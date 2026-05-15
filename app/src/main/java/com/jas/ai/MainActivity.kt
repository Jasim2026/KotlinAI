package com.jas.ai

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.util.JsonReader
import android.util.Log
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
import java.io.FileOutputStream
import java.io.FileReader
import java.util.UUID

/**
 * MODELS STATE
 */
enum class ModelState { CHECKING, NEEDS_MAIN_MODEL, NEEDS_EMBEDDING_MODEL, NEEDS_FAISS_DB, LOADING_DB, COPYING, READY, ERROR }

data class ChatMessage(val id: String = UUID.randomUUID().toString(), val text: String, val isFromUser: Boolean, val isLoading: Boolean = false)

/**
 * VECTOR SEARCH ENGINE WITH KEYWORD FALLBACK
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
                if (text.isNotEmpty() && vector != null) records.add(Record(text, vector))
            }
            reader.endArray()
        }
    }

    fun searchVector(query: FloatArray, topK: Int = 3): List<String> {
        if (records.isEmpty()) return emptyList()
        return records.map { it.text to cosineSimilarity(query, it.vector) }
            .sortedByDescending { it.second }.take(topK).map { it.first }
    }

    fun searchKeyword(query: String, topK: Int = 3): List<String> {
        val queryWords = query.lowercase().split(" ").filter { it.length > 3 }.toSet()
        if (queryWords.isEmpty()) return records.take(topK).map { it.text }
        return records.map { record ->
            val score = queryWords.count { record.text.lowercase().contains(it) }
            record.text to score
        }.filter { it.second > 0 }.sortedByDescending { it.second }.take(topK).map { it.first }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var nA = 0f; var nB = 0f
        for (i in a.indices) { dot += a[i] * b[i]; nA += a[i] * a[i]; nB += b[i] * b[i] }
        val res = dot / (kotlin.math.sqrt(nA) * kotlin.math.sqrt(nB))
        return if (res.isNaN()) 0f else res
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
    private var useKeywordFallback = false

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
                    _modelState.value = ModelState.LOADING_DB
                    vectorDB.loadFromFile(faiss)
                    initializeEngines(context, main.absolutePath, embed.absolutePath)
                }
            }
        }
    }

    fun copyFile(context: Context, uri: Uri, name: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _modelState.value = ModelState.COPYING
            try {
                context.contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(File(context.filesDir, name)).use { output ->
                        input.copyTo(output) // SAFE STREAMING COPY
                    }
                }
                checkState(context)
            } catch (t: Throwable) {
                _errorMessage.value = "Copy Error: ${t.message}"; _modelState.value = ModelState.ERROR
            }
        }
    }

    private fun initializeEngines(context: Context, mainPath: String, embedPath: String) {
        try {
            Engine.setNativeMinLogSeverity(LogSeverity.ERROR)
            mainEngine = Engine(EngineConfig(modelPath = mainPath))
            mainEngine?.initialize()
            mainConversation = mainEngine?.createConversation()

            try {
                val pfd = ParcelFileDescriptor.open(File(embedPath), ParcelFileDescriptor.MODE_READ_ONLY)
                val options = TextEmbedder.TextEmbedderOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetFileDescriptor(pfd.fd).build())
                    .build()
                textEmbedder = TextEmbedder.createFromOptions(context, options)
                useKeywordFallback = false
            } catch (e: Exception) {
                useKeywordFallback = true
            }

            _messages.value = listOf(ChatMessage(text = "System Ready. Mode: ${if(useKeywordFallback) "Keyword" else "Neural"}", isFromUser = false))
            _modelState.value = ModelState.READY
        } catch (e: Exception) {
            _errorMessage.value = "Engine Error: ${e.message}"; _modelState.value = ModelState.ERROR
        }
    }

    fun sendMessage(prompt: String) {
        if (prompt.isBlank()) return
        _messages.value += ChatMessage(text = prompt, isFromUser = true)
        val botMsgId = UUID.randomUUID().toString()
        _messages.value += ChatMessage(id = botMsgId, text = "", isFromUser = false, isLoading = true)

        viewModelScope.launch(Dispatchers.IO) {
            try {
                updateBotStatus(botMsgId, "Analyzing...", true)
                val decision = querySync("SYSTEM: Reply RAG_REQUIRED if the user asks for facts, else reply DIRECT. USER: $prompt")
                if (decision.contains("RAG_REQUIRED", ignoreCase = true)) {
                    updateBotStatus(botMsgId, "Retrieving Local Context...", true)
                    val contextList = if (useKeywordFallback || textEmbedder == null) {
                        vectorDB.searchKeyword(prompt)
                    } else {
                        try {
                            val vector = textEmbedder!!.embed(prompt).embeddingResult().embeddings().first().floatEmbedding()
                            vectorDB.searchVector(vector)
                        } catch (e: Exception) { vectorDB.searchKeyword(prompt) }
                    }
                    val contextString = if(contextList.isEmpty()) "No local data found." else contextList.joinToString("\n---\n")
                    streamResponse(botMsgId, "CONTEXT:\n$contextString\n\nUSER QUESTION: $prompt")
                } else {
                    streamResponse(botMsgId, prompt)
                }
            } catch (e: Exception) { updateBotStatus(botMsgId, "Error: ${e.message}", false) }
        }
    }

    private suspend fun querySync(p: String): String {
        val conv = mainEngine?.createConversation() ?: return "DIRECT"
        var res = ""
        try { conv.sendMessageAsync(p).collect { res += it } } finally { conv.close() }
        return res
    }

    private suspend fun streamResponse(msgId: String, p: String) {
        var full = ""
        mainConversation?.sendMessageAsync(p)?.collect { full += it; updateBotStatus(msgId, full, true) }
        updateBotStatus(msgId, full, false)
    }

    private fun updateBotStatus(id: String, text: String, loading: Boolean) {
        val list = _messages.value.toMutableList()
        val i = list.indexOfFirst { it.id == id }
        if (i != -1) { list[i] = list[i].copy(text = text, isLoading = loading); _messages.value = list }
    }

    fun reset(context: Context) {
        viewModelScope.launch(Dispatchers.IO) {
            mainConversation?.close(); mainEngine?.close(); textEmbedder?.close()
            File(context.filesDir, "main_model.litertlm").delete()
            File(context.filesDir, "embedding_model.litertlm").delete()
            File(context.filesDir, "faiss_db.json").delete()
            _messages.value = emptyList(); _modelState.value = ModelState.CHECKING; checkState(context)
        }
    }
}

/**
 * UI
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { AppTheme { Surface(Modifier.fillMaxSize()) { MainScreen() } } }
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
        topBar = { TopAppBar(title = { Text("Agentic RAG", fontWeight = FontWeight.Bold) }, 
            actions = { if(state == ModelState.READY) TextButton(onClick={viewModel.reset(context)}){ Text("Reset", color=Color.Red) } }) },
        bottomBar = { if(state == ModelState.READY) ChatInputBar { viewModel.sendMessage(it) } }
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize()) {
            when(state) {
                ModelState.CHECKING -> LoadingUI("Checking Storage...")
                ModelState.NEEDS_MAIN_MODEL -> SetupUI("1. Select Gemma-1B/2B", "Select File") { p1.launch(arrayOf("*/*")) }
                ModelState.NEEDS_EMBEDDING_MODEL -> SetupUI("2. Select Embedding Model", "Select File") { p2.launch(arrayOf("*/*")) }
                ModelState.NEEDS_FAISS_DB -> SetupUI("3. Select Knowledge Base", "Select JSON") { p3.launch(arrayOf("*/*")) }
                ModelState.COPYING -> LoadingUI("Saving Models to App Storage...")
                ModelState.LOADING_DB -> LoadingUI("Starting AI Engines...")
                ModelState.ERROR -> SetupUI("Critical Error:\n$error", "Reset App") { viewModel.reset(context) }
                ModelState.READY -> ChatList(messages)
            }
        }
    }
}

@Composable
fun LoadingUI(m: String) {
    Column(Modifier.fillMaxSize(), Arrangement.Center, Alignment.CenterHorizontally) { CircularProgressIndicator(); Spacer(Modifier.height(16.dp)); Text(m) }
}

@Composable
fun SetupUI(t: String, b: String, onClick: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(32.dp), Arrangement.Center, Alignment.CenterHorizontally) {
        Text(t, style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center); Spacer(Modifier.height(20.dp)); Button(onClick) { Text(b) }
    }
}

@Composable
fun ChatList(msgs: List<ChatMessage>) {
    val listState = rememberLazyListState()
    LaunchedEffect(msgs.size) { if(msgs.isNotEmpty()) listState.animateScrollToItem(msgs.lastIndex) }
    LazyColumn(state = listState, modifier = Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp)) {
        items(msgs) { msg ->
            val align = if (msg.isFromUser) Alignment.CenterEnd else Alignment.CenterStart
            val color = if (msg.isFromUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
            Box(Modifier.fillMaxWidth().padding(4.dp), contentAlignment = align) {
                Surface(color = color, shape = RoundedCornerShape(12.dp)) {
                    Text(msg.text + (if(msg.isLoading) "..." else ""), Modifier.padding(12.dp))
                }
            }
        }
    }
}

@Composable
fun ChatInputBar(onSend: (String) -> Unit) {
    var text by remember { mutableStateOf("") }
    Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(text, { text = it }, Modifier.weight(1f), shape = RoundedCornerShape(24.dp))
        IconButton(onClick = { if(text.isNotBlank()){ onSend(text); text="" } }) { Icon(Icons.Default.Send, null) }
    }
}