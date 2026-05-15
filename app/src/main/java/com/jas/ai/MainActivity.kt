package com.jas.ai

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.util.JsonReader
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.LogSeverity
import com.jas.ai.ui.theme.AppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.text.embedder.TextEmbedder
import java.io.File
import java.io.FileOutputStream
import java.io.FileReader
import java.util.UUID

/**
 * MODELS STATE ENUM
 * Orchestrates the multi-step initialization process
 */
enum class ModelState {
    CHECKING,
    NEEDS_MAIN_MODEL,
    NEEDS_EMBEDDING_MODEL,
    NEEDS_FAISS_DB,
    LOADING_DB,
    COPYING,
    READY,
    ERROR
}

/**
 * CHAT DATA MODEL
 */
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val isFromUser: Boolean,
    val isLoading: Boolean = false
)

/**
 * VECTOR SEARCH ENGINE
 * Optimized for local memory using streaming JSON parsing
 */
class LocalVectorDB {
    data class Record(val text: String, val vector: FloatArray)
    private val records = mutableListOf<Record>()

    fun loadFromFile(dbFile: File) {
        records.clear()
        if (!dbFile.exists()) return
        
        JsonReader(FileReader(dbFile)).use { reader ->
            reader.beginArray()
            while (reader.hasNext()) {
                var text = ""
                var vector: FloatArray? = null
                reader.beginObject()
                while (reader.hasNext()) {
                    when (reader.nextName()) {
                        "text" -> text = reader.nextString()
                        "vector" -> {
                            val list = mutableListOf<Float>()
                            reader.beginArray()
                            while (reader.hasNext()) list.add(reader.nextDouble().toFloat())
                            reader.endArray()
                            vector = list.toFloatArray()
                        }
                        else -> reader.skipValue()
                    }
                }
                reader.endObject()
                if (text.isNotEmpty() && vector != null) {
                    records.add(Record(text, vector))
                }
            }
            reader.endArray()
        }
    }

    fun search(query: FloatArray, topK: Int = 3): List<String> {
        if (records.isEmpty()) return emptyList()
        return records.map { it.text to cosineSimilarity(query, it.vector) }
            .sortedByDescending { it.second }
            .take(topK)
            .map { it.first }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        return dotProduct / (kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB))
    }

    fun clear() = records.clear()
}

/**
 * VIEWMODEL - AGENTIC PIPELINE ORCHESTRATOR
 */
class ChatViewModel : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _modelState = MutableStateFlow(ModelState.CHECKING)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    private val _errorMessage = MutableStateFlow("")
    val errorMessage: StateFlow<String> = _errorMessage.asStateFlow()

    // LiteRT-LM (Gemma 1B)
    private var mainEngine: Engine? = null
    private var mainConversation: Conversation? = null
    
    // TFLite Task Library (Embedding Gemma 300M)
    private var textEmbedder: TextEmbedder? = null
    
    private val vectorDB = LocalVectorDB()

    fun checkState(context: Context) {
        viewModelScope.launch(Dispatchers.IO) {
            val main = File(context.filesDir, "main_model.litertlm")
            val embed = File(context.filesDir, "embedding_model.litertlm")
            val faiss = File(context.filesDir, "faiss_db.json")

            when {
                !main.exists() -> _modelState.value = ModelState.NEEDS_MAIN_MODEL
                !embed.exists() -> _modelState.value = ModelState.NEEDS_EMBEDDING_MODEL
                !faiss.exists() -> _modelState.value = ModelState.NEEDS_FAISS_DB
                else -> {
                    try {
                        _modelState.value = ModelState.LOADING_DB
                        vectorDB.loadFromFile(faiss)
                        initializeEngines(main.absolutePath, embed.absolutePath)
                    } catch (t: Throwable) {
                        _errorMessage.value = "Init Failed: ${t.message}"
                        _modelState.value = ModelState.ERROR
                    }
                }
            }
        }
    }

    fun copyFile(context: Context, uri: Uri, name: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _modelState.value = ModelState.COPYING
            try {
                context.contentResolver.openInputStream(uri)?.use { input ->
                    val file = File(context.filesDir, name)
                    FileOutputStream(file).use { input.copyTo(it) }
                }
                checkState(context)
            } catch (t: Throwable) {
                _errorMessage.value = "Copy Failed: ${t.message}"
                _modelState.value = ModelState.ERROR
            }
        }
    }

    private fun initializeEngines(mainPath: String, embedPath: String) {
        try {
            // 1. Initialize Generative Engine
            Engine.setNativeMinLogSeverity(LogSeverity.ERROR)
            mainEngine = Engine(EngineConfig(modelPath = mainPath)).apply { initialize() }
            mainConversation = mainEngine?.createConversation()

            // 2. Initialize Embedding Engine
            val baseOptions = BaseOptions.builder().build()
            val embedderOptions = TextEmbedder.TextEmbedderOptions.builder()
                .setBaseOptions(baseOptions)
                .build()
            textEmbedder = TextEmbedder.createFromFileAndOptions(File(embedPath), embedderOptions)

            _messages.value = listOf(ChatMessage("Local Agent Active. RAG-ready.", false))
            _modelState.value = ModelState.READY
        } catch (e: Exception) {
            _errorMessage.value = "Engine Error: ${e.message}"
            _modelState.value = ModelState.ERROR
        }
    }

    fun sendMessage(text: String) {
        if (text.isBlank()) return
        _messages.value += ChatMessage(text, true)
        
        val agentMsgId = UUID.randomUUID().toString()
        _messages.value += ChatMessage(id = agentMsgId, text = "", isFromUser = false, isLoading = true)

        viewModelScope.launch(Dispatchers.IO) {
            try {
                // PHASE 1: AGENTIC ROUTING
                updateBotStatus(agentMsgId, "Agent deciding if context is needed...", true)
                val routerPrompt = """
                    SYSTEM: You are a routing agent.
                    If the user query requires external knowledge or facts, reply ONLY with 'RAG_REQUIRED'.
                    If it is a general chat or greeting, reply ONLY with 'DIRECT_ANSWER'.
                    USER: $text
                """.trimIndent()
                
                val decision = querySync(routerPrompt)
                val isRag = decision.contains("RAG_REQUIRED", ignoreCase = true)

                if (isRag) {
                    // PHASE 2: RETRIEVAL
                    updateBotStatus(agentMsgId, "RAG Triggered. Embedding query...", true)
                    val vector = generateEmbed(text)
                    
                    updateBotStatus(agentMsgId, "Searching local Faiss DB...", true)
                    val contextList = vectorDB.search(vector)
                    val contextString = contextList.joinToString("\n---\n")

                    // PHASE 3: AUGMENTED GENERATION
                    updateBotStatus(agentMsgId, "Synthesizing answer from context...", true)
                    val ragPrompt = """
                        SYSTEM: Answer using only the provided context.
                        CONTEXT: $contextString
                        USER: $text
                    """.trimIndent()
                    streamResponse(agentMsgId, ragPrompt)
                } else {
                    // PHASE 2 (ALTERNATIVE): DIRECT ANSWER
                    updateBotStatus(agentMsgId, "Direct answer chosen...", true)
                    streamResponse(agentMsgId, text)
                }
            } catch (e: Exception) {
                updateBotStatus(agentMsgId, "Pipeline Error: ${e.message}", false)
            }
        }
    }

    private suspend fun querySync(p: String): String {
        val conv = mainEngine?.createConversation() ?: return ""
        var res = ""
        conv.sendMessageAsync(p).collect { res += it }
        conv.close()
        return res.trim()
    }

    private suspend fun streamResponse(msgId: String, p: String) {
        var fullText = ""
        mainConversation?.sendMessageAsync(p)?.collect { token ->
            fullText += token
            updateBotStatus(msgId, fullText, true)
        }
        updateBotStatus(msgId, fullText, false)
    }

    private fun generateEmbed(text: String): FloatArray {
        val embedder = textEmbedder ?: throw Exception("Embedder not initialized")
        val result = embedder.embed(text)
        return result.embeddingResult.embeddings.first().floatArray
    }

    private fun updateBotStatus(id: String, text: String, loading: Boolean) {
        val currentList = _messages.value.toMutableList()
        val index = currentList.indexOfFirst { it.id == id }
        if (index != -1) {
            currentList[index] = currentList[index].copy(text = text, isLoading = loading)
            _messages.value = currentList
        }
    }

    fun reset(context: Context) {
        viewModelScope.launch(Dispatchers.IO) {
            mainConversation?.close()
            mainEngine?.close()
            textEmbedder?.close()
            File(context.filesDir, "main_model.litertlm").delete()
            File(context.filesDir, "embedding_model.litertlm").delete()
            File(context.filesDir, "faiss_db.json").delete()
            vectorDB.clear()
            _messages.value = emptyList()
            _modelState.value = ModelState.CHECKING
            checkState(context)
        }
    }

    override fun onCleared() {
        super.onCleared()
        mainConversation?.close()
        mainEngine?.close()
        textEmbedder?.close()
    }
}

/**
 * MAIN ACTIVITY & UI
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AppTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    MainScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(viewModel: ChatViewModel = viewModel()) {
    val state by viewModel.modelState.collectAsState()
    val error by viewModel.errorMessage.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val context = LocalContext.current

    LaunchedEffect(Unit) { viewModel.checkState(context) }

    val mainPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        it?.let { viewModel.copyFile(context, it, "main_model.litertlm") }
    }
    val embedPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        it?.let { viewModel.copyFile(context, it, "embedding_model.litertlm") }
    }
    val jsonPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        it?.let { viewModel.copyFile(context, it, "faiss_db.json") }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Agentic Kotlin RAG", fontWeight = FontWeight.ExtraBold) },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                actions = {
                    if (state == ModelState.READY) {
                        IconButton(onClick = { viewModel.reset(context) }) {
                            Text("Reset", color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            )
        },
        bottomBar = {
            if (state == ModelState.READY) {
                ChatInputBar { viewModel.sendMessage(it) }
            }
        }
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize()) {
            when (state) {
                ModelState.CHECKING -> LoadingUI("Scanning Internal Storage...")
                ModelState.NEEDS_MAIN_MODEL -> SetupUI("Step 1: Main Model Required", "Select gemma-1b.litertlm") {
                    mainPicker.launch(arrayOf("*/*"))
                }
                ModelState.NEEDS_EMBEDDING_MODEL -> SetupUI("Step 2: Embedding Model Required", "Select embedding-300m.litertlm") {
                    embedPicker.launch(arrayOf("*/*"))
                }
                ModelState.NEEDS_FAISS_DB -> SetupUI("Step 3: Vector Knowledge Required", "Select faiss_db.json") {
                    jsonPicker.launch(arrayOf("application/json", "*/*"))
                }
                ModelState.COPYING -> LoadingUI("Securing file in app sandbox...")
                ModelState.LOADING_DB -> LoadingUI("Streaming Vector DB into RAM...")
                ModelState.ERROR -> SetupUI("Critical Error Occurred", "Fix & Retry: $error") {
                    viewModel.checkState(context)
                }
                ModelState.READY -> ChatList(messages)
            }
        }
    }
}

@Composable
fun LoadingUI(m: String) {
    Column(Modifier.fillMaxSize(), Arrangement.Center, Alignment.CenterHorizontally) {
        CircularProgressIndicator(Modifier.size(60.dp))
        Spacer(Modifier.height(20.dp))
        Text(m, style = MaterialTheme.typography.bodyLarge)
    }
}

@Composable
fun SetupUI(title: String, btn: String, onClick: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(40.dp), Arrangement.Center, Alignment.CenterHorizontally) {
        Text(title, style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center)
        Spacer(Modifier.height(30.dp))
        Button(onClick, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth().height(60.dp)) {
            Text(btn)
        }
    }
}

@Composable
fun ChatList(messages: List<ChatMessage>) {
    val scrollState = rememberLazyListState()
    LaunchedEffect(messages.size) { if(messages.isNotEmpty()) scrollState.animateScrollToItem(messages.lastIndex) }

    LazyColumn(state = scrollState, modifier = Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp)) {
        items(messages) { ChatBubble(it) }
    }
}

@Composable
fun ChatBubble(msg: ChatMessage) {
    val alignment = if (msg.isFromUser) Alignment.CenterEnd else Alignment.CenterStart
    val color = if (msg.isFromUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
    val textColor = if (msg.isFromUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant
    val shape = if (msg.isFromUser) RoundedCornerShape(16.dp, 16.dp, 2.dp, 16.dp) else RoundedCornerShape(16.dp, 16.dp, 16.dp, 2.dp)

    Box(Modifier.fillMaxWidth().padding(vertical = 4.dp), contentAlignment = alignment) {
        Column(horizontalAlignment = if(msg.isFromUser) Alignment.End else Alignment.Start) {
            Surface(color = color, shape = shape, shadowElevation = 2.dp) {
                Text(msg.text, modifier = Modifier.padding(12.dp), color = textColor, style = MaterialTheme.typography.bodyLarge)
            }
            if (msg.isLoading) {
                Text("Processing...", style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(4.dp), color = Color.Gray)
            }
        }
    }
}

@Composable
fun ChatInputBar(onSend: (String) -> Unit) {
    var text by remember { mutableStateOf("") }
    Surface(tonalElevation = 8.dp) {
        Row(Modifier.fillMaxWidth().padding(12.dp).navigationBarsPadding().imePadding(), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Ask the local agent...") },
                shape = RoundedCornerShape(24.dp),
                keyboardOptions = KeyboardOptions(KeyboardCapitalization.Sentences)
            )
            Spacer(Modifier.width(8.dp))
            FloatingActionButton(onClick = { if(text.isNotBlank()){ onSend(text); text="" } }, shape = RoundedCornerShape(50.dp)) {
                Icon(Icons.Default.Send, null)
            }
        }
    }
}