package com.jas.ai

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.util.JsonReader
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.LogSeverity
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import com.jas.ai.ui.theme.AppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.FileReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

/**
 * MODELS STATE
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
 * CHAT MESSAGE DATA CLASS
 */
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val isFromUser: Boolean,
    val isLoading: Boolean = false
)

/**
 * VECTOR SEARCH ENGINE
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
                            while (reader.hasNext()) {
                                list.add(reader.nextDouble().toFloat())
                            }
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
        val result = dotProduct / (kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB))
        return if (result.isNaN()) 0f else result
    }

    fun clear() = records.clear()
}

/**
 * VIEWMODEL
 */
class ChatViewModel : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _modelState = MutableStateFlow(ModelState.CHECKING)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    private val _errorMessage = MutableStateFlow("")
    val errorMessage: StateFlow<String> = _errorMessage.asStateFlow()

    private var mainEngine: Engine? = null
    private var mainConversation: Conversation? = null
    private var textEmbedder: TextEmbedder? = null
    private val vectorDB = LocalVectorDB()

    // CRITICAL: Prevent Garbage Collection of the model buffer during operation
    private var embeddingBuffer: ByteBuffer? = null

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
                        initializeEngines(context, main.absolutePath, embed.absolutePath)
                    } catch (t: Throwable) {
                        _errorMessage.value = t.message ?: "Initialization Failed"
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
                    FileOutputStream(File(context.filesDir, name)).use { input.copyTo(it) }
                }
                checkState(context)
            } catch (t: Throwable) {
                _errorMessage.value = "Copy Failed: ${t.message}"
                _modelState.value = ModelState.ERROR
            }
        }
    }

    private fun initializeEngines(context: Context, mainPath: String, embedPath: String) {
        // 1. Initialize Generative Engine (Gemma 1B/2B)
        try {
            Engine.setNativeMinLogSeverity(LogSeverity.ERROR)
            mainEngine = Engine(EngineConfig(modelPath = mainPath))
            mainEngine?.initialize()
            mainConversation = mainEngine?.createConversation()
        } catch (e: Exception) {
            throw Exception("Main Engine Error: ${e.message}")
        }

        // 2. Initialize Embedding Engine (Embedding Gemma 300M)
        try {
            val file = File(embedPath)
            val inputStream = FileInputStream(file)
            val channel = inputStream.channel
            
            // NON-NEGOTIABLE FIX: Load to Direct ByteBuffer to bypass mmap allocation limits
            embeddingBuffer = ByteBuffer.allocateDirect(file.length().toInt()).apply {
                order(ByteOrder.nativeOrder())
                channel.read(this)
                flip()
            }
            channel.close()
            inputStream.close()

            val baseOptions = BaseOptions.builder()
                .setModelAssetBuffer(embeddingBuffer)
                .build()
                
            val options = TextEmbedder.TextEmbedderOptions.builder()
                .setBaseOptions(baseOptions)
                .build()
            
            textEmbedder = TextEmbedder.createFromOptions(context, options)

            _messages.value = listOf(ChatMessage(text = "System Ready. Agentic RAG Active.", isFromUser = false))
            _modelState.value = ModelState.READY
        } catch (e: Exception) {
            throw Exception("Embedding Engine Error: ${e.message}")
        }
    }

    fun sendMessage(prompt: String) {
        if (prompt.isBlank()) return
        _messages.value += ChatMessage(text = prompt, isFromUser = true)
        
        val botMsgId = UUID.randomUUID().toString()
        _messages.value += ChatMessage(id = botMsgId, text = "", isFromUser = false, isLoading = true)

        viewModelScope.launch(Dispatchers.IO) {
            try {
                updateBotStatus(botMsgId, "Agent analyzing query...", true)
                
                val routePrompt = "SYSTEM: Reply RAG_REQUIRED if this query needs facts, else reply DIRECT. USER: $prompt"
                val decision = querySync(routePrompt)

                if (decision.contains("RAG_REQUIRED", ignoreCase = true)) {
                    updateBotStatus(botMsgId, "Retrieving context from Vector DB...", true)
                    val vector = generateEmbed(prompt)
                    val contextList = vectorDB.search(vector)
                    val contextString = contextList.joinToString("\n---\n")

                    updateBotStatus(botMsgId, "Generating answer with RAG...", true)
                    val finalPrompt = "CONTEXT: $contextString\nUSER: $prompt"
                    streamResponse(botMsgId, finalPrompt)
                } else {
                    updateBotStatus(botMsgId, "Generating direct answer...", true)
                    streamResponse(botMsgId, prompt)
                }
            } catch (e: Exception) {
                updateBotStatus(botMsgId, "Pipeline Error: ${e.message}", false)
            }
        }
    }

    private suspend fun querySync(p: String): String {
        val conv = mainEngine?.createConversation() ?: return ""
        var result = ""
        try {
            conv.sendMessageAsync(p).collect { result += it }
        } finally {
            conv.close()
        }
        return result.trim()
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
        val embedder = textEmbedder ?: throw Exception("Embedder is null")
        val results = embedder.embed(text)
        return results.embeddingResult().embeddings().first().floatEmbedding()
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
            try { mainConversation?.close() } catch (e: Exception) {}
            try { mainEngine?.close() } catch (e: Exception) {}
            try { textEmbedder?.close() } catch (e: Exception) {}
            embeddingBuffer = null
            
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
        embeddingBuffer = null
    }
}

/**
 * UI COMPONENTS
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

    val p1 = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { it?.let { viewModel.copyFile(context, it, "main_model.litertlm") } }
    val p2 = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { it?.let { viewModel.copyFile(context, it, "embedding_model.litertlm") } }
    val p3 = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { it?.let { viewModel.copyFile(context, it, "faiss_db.json") } }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Local Agentic RAG", fontWeight = FontWeight.Bold) },
                actions = {
                    if (state == ModelState.READY) {
                        Button(onClick = { viewModel.reset(context) }) { Text("Reset") }
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
                ModelState.CHECKING -> LoadingUI("Scanning storage...")
                ModelState.NEEDS_MAIN_MODEL -> SetupUI("1. Main Model Required", "Select Gemma-1B/2B (.bin)") { p1.launch(arrayOf("*/*")) }
                ModelState.NEEDS_EMBEDDING_MODEL -> SetupUI("2. Embedding Model Required", "Select Embedding-300M (.tflite)") { p2.launch(arrayOf("*/*")) }
                ModelState.NEEDS_FAISS_DB -> SetupUI("3. Vector DB Required", "Select faiss_db.json") { p3.launch(arrayOf("*/*")) }
                ModelState.COPYING -> LoadingUI("Copying files to app sandbox...")
                ModelState.LOADING_DB -> LoadingUI("Loading Vector DB...")
                ModelState.ERROR -> SetupUI("Error:\n$error", "Reset & Try Again") { viewModel.reset(context) }
                ModelState.READY -> ChatList(messages)
            }
        }
    }
}

@Composable
fun LoadingUI(m: String) {
    Column(Modifier.fillMaxSize(), Arrangement.Center, Alignment.CenterHorizontally) {
        CircularProgressIndicator()
        Spacer(Modifier.height(16.dp))
        Text(m)
    }
}

@Composable
fun SetupUI(t: String, b: String, onClick: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(32.dp), Arrangement.Center, Alignment.CenterHorizontally) {
        Text(t, style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center)
        Spacer(Modifier.height(24.dp))
        Button(onClick, Modifier.fillMaxWidth().height(56.dp)) { Text(b) }
    }
}

@Composable
fun ChatList(msgs: List<ChatMessage>) {
    val listState = rememberLazyListState()
    LaunchedEffect(msgs.size) { if(msgs.isNotEmpty()) listState.animateScrollToItem(msgs.lastIndex) }
    LazyColumn(state = listState, modifier = Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp)) {
        items(msgs) { ChatBubble(it) }
    }
}

@Composable
fun ChatBubble(msg: ChatMessage) {
    val align = if (msg.isFromUser) Alignment.CenterEnd else Alignment.CenterStart
    val color = if (msg.isFromUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
    Box(Modifier.fillMaxWidth().padding(vertical = 4.dp), contentAlignment = align) {
        Surface(color = color, shape = RoundedCornerShape(12.dp)) {
            Text(msg.text + (if(msg.isLoading) "..." else ""), Modifier.padding(12.dp))
        }
    }
}

@Composable
fun ChatInputBar(onSend: (String) -> Unit) {
    var text by remember { mutableStateOf("") }
    Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(text, { text = it }, Modifier.weight(1f), shape = RoundedCornerShape(24.dp))
        Spacer(Modifier.width(8.dp))
        IconButton(onClick = { if(text.isNotBlank()){ onSend(text); text="" } }) { Icon(Icons.Default.Send, null) }
    }
}