import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqlite3/sqlite3.dart';

class DatabaseService {
  static final DatabaseService instance = DatabaseService._init();
  static Database? _database;

  DatabaseService._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB('native_llama.db');
    return _database!;
  }

  Future<Database> _initDB(String fileName) async {
    final documentsDirectory = await getApplicationDocumentsDirectory();
    final path = join(documentsDirectory.path, fileName);
    final db = sqlite3.open(path);

    db.execute('''
      CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        createdAt TEXT NOT NULL
      )
    ''');

    db.execute('''
      CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sessionId INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        createdAt TEXT NOT NULL,
        mediaPaths TEXT,
        mediaType TEXT,
        FOREIGN KEY (sessionId) REFERENCES chat_sessions (id) ON DELETE CASCADE
      )
    ''');

    // Simple migration: Check if mediaPaths column exists, if not add it
    final columns = db.select("PRAGMA table_info(chat_messages)");
    bool hasMediaPaths = columns.any((column) => column['name'] == 'mediaPaths');
    if (!hasMediaPaths) {
      db.execute("ALTER TABLE chat_messages ADD COLUMN mediaPaths TEXT");
      db.execute("ALTER TABLE chat_messages ADD COLUMN mediaType TEXT");
    }

    db.execute('''
      CREATE TABLE IF NOT EXISTS semantic_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        response TEXT NOT NULL,
        embedding TEXT NOT NULL,
        createdAt TEXT NOT NULL
      )
    ''');

    return db;
  }

  Future<void> saveToCache(String query, String response, List<double> embedding) async {
    final db = await database;
    final stmt = db.prepare('INSERT INTO semantic_cache (query, response, embedding, createdAt) VALUES (?, ?, ?, ?)');
    stmt.execute([query, response, embedding.join(','), DateTime.now().toIso8601String()]);
    stmt.dispose();
  }

  Future<List<Map<String, dynamic>>> getCacheEntries() async {
    final db = await database;
    final results = db.select('SELECT * FROM semantic_cache');
    return results.map((row) => {
      'query': row['query'],
      'response': row['response'],
      'embedding': (row['embedding'] as String).split(',').map(double.parse).toList(),
    }).toList();
  }

  Future<int> createChatSession(String title) async {
    final db = await database;
    final stmt = db.prepare('INSERT INTO chat_sessions (title, createdAt) VALUES (?, ?)');
    stmt.execute([title, DateTime.now().toIso8601String()]);
    stmt.dispose();
    return db.lastInsertRowId;
  }

  Future<List<Map<String, dynamic>>> getChatSessions() async {
    final db = await database;
    final results = db.select('SELECT * FROM chat_sessions ORDER BY createdAt DESC');
    return results.map((row) => {'id': row['id'], 'title': row['title'], 'createdAt': row['createdAt']}).toList();
  }

  Future<void> deleteChatSession(int id) async {
    final db = await database;
    db.execute('DELETE FROM chat_sessions WHERE id = ?', [id]);
  }

  Future<void> saveChatMessage(int sessionId, String role, String content, {List<String>? mediaPaths, String? mediaType}) async {
    final db = await database;
    final stmt = db.prepare('INSERT INTO chat_messages (sessionId, role, content, createdAt, mediaPaths, mediaType) VALUES (?, ?, ?, ?, ?, ?)');
    stmt.execute([
      sessionId,
      role,
      content,
      DateTime.now().toIso8601String(),
      mediaPaths?.join(','),
      mediaType
    ]);
    stmt.dispose();
  }

  Future<List<Map<String, dynamic>>> getChatMessages(int sessionId) async {
    final db = await database;
    final results = db.select('SELECT * FROM chat_messages WHERE sessionId = ? ORDER BY createdAt ASC', [sessionId]);
    return results.map((row) => {
      'id': row['id'],
      'sessionId': row['sessionId'],
      'role': row['role'],
      'content': row['content'],
      'createdAt': row['createdAt'],
      'mediaPaths': row['mediaPaths'],
      'mediaType': row['mediaType'],
    }).toList();
  }
}