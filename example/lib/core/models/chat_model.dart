class ChatSession {
  final int? id;
  final String title;
  final DateTime createdAt;

  ChatSession({
    this.id,
    required this.title,
    required this.createdAt,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'title': title,
      'createdAt': createdAt.toIso8601String(),
    };
  }

  factory ChatSession.fromMap(Map<String, dynamic> map) {
    return ChatSession(
      id: map['id'],
      title: map['title'],
      createdAt: DateTime.parse(map['createdAt']),
    );
  }
}

class ChatMessage {
  final int? id;
  final int sessionId;
  final String role;
  final String content;
  final DateTime createdAt;
  final List<String>? mediaPaths;
  final String? mediaType;
  
  // UI-only properties (not stored in DB currently, or can be parsed from content)
  String thoughtText;
  String answerText;
  int? thoughtDuration;
  bool isSearching;
  List<Map<String, String>> searchResults;
  String? searchStatus;
  final bool isFinalized;
  final bool expectTags;

  ChatMessage({
    this.id,
    required this.sessionId,
    required this.role,
    required this.content,
    required this.createdAt,
    this.mediaPaths,
    this.mediaType,
    this.thoughtText = "",
    this.answerText = "",
    this.thoughtDuration,
    this.isSearching = false,
    this.searchResults = const [],
    this.searchStatus,
    this.isFinalized = false,
    this.expectTags = false,
  }) {
    if (role == 'assistant') {
      _parseContent(content, isFinalized);
    } else {
      answerText = content;
    }
  }

  void updateContent(String newContent, {bool isFinalized = false}) {
    // This is used for streaming
    _parseContent(newContent, isFinalized);
  }

  void _parseContent([String? overrideContent, bool isFinalized = false]) {
    final String text = (overrideContent ?? content).trim();
    
    if (text.isEmpty) {
      thoughtText = "";
      answerText = "";
      return;
    }

    // Heuristic for partial tags during streaming (e.g., "<thou")
    if (!isFinalized && text.startsWith('<') && !text.contains('>')) {
      thoughtText = "";
      answerText = "";
      return;
    }

    int thoughtStart = text.indexOf('<thought>');
    int thoughtEnd = text.indexOf('</thought>');
    int answerStart = text.indexOf('<final_answer>');
    int answerEnd = text.indexOf('</final_answer>');

    String currentThought = "";
    String currentAnswer = "";

    if (thoughtStart != -1) {
      // Preamble before the first tag is considered part of the thought
      String preamble = text.substring(0, thoughtStart).trim();
      
      if (thoughtEnd != -1) {
        currentThought = (preamble.isNotEmpty ? preamble + "\n" : "") + 
                         text.substring(thoughtStart + '<thought>'.length, thoughtEnd).trim();
        
        String remaining = text.substring(thoughtEnd + '</thought>'.length);
        
        // Look for answer tag in remaining text
        int subAnswerStart = remaining.indexOf('<final_answer>');
        if (subAnswerStart != -1) {
          // Anything between </thought> and <final_answer> is also thought
          String interText = remaining.substring(0, subAnswerStart).trim();
          if (interText.isNotEmpty) currentThought += "\n" + interText;

          int subAnswerEnd = remaining.indexOf('</final_answer>');
          currentAnswer = subAnswerEnd != -1
              ? remaining.substring(subAnswerStart + '<final_answer>'.length, subAnswerEnd).trim()
              : remaining.substring(subAnswerStart + '<final_answer>'.length).trim();
        } else {
          // No <final_answer> yet after </thought>
          if (isFinalized) {
            // Finished, so everything after </thought> is the answer fallback
            currentAnswer = remaining.trim();
          } else {
            // Still streaming, keep everything after </thought> in thought until we see <final_answer>
            if (remaining.trim().isNotEmpty) currentThought += "\n" + remaining.trim();
          }
        }
      } else {
        // Open thought tag - everything after preamble + <thought> is in currentThought
        currentThought = (preamble.isNotEmpty ? preamble + "\n" : "") + 
                        text.substring(thoughtStart + '<thought>'.length).trim();
      }
    } else if (answerStart != -1) {
      // No <thought> tag, but has <final_answer> tag. preamble is thought.
      currentThought = text.substring(0, answerStart).trim();
      
      int subAnswerEnd = text.indexOf('</final_answer>');
      currentAnswer = subAnswerEnd != -1
          ? text.substring(answerStart + '<final_answer>'.length, subAnswerEnd).trim()
          : text.substring(answerStart + '<final_answer>'.length).trim();
    } else {
      // No tags at all.
      if (isFinalized) {
        currentAnswer = text;
        currentThought = "";
      } else {
        // While streaming, if we expect tags, assume untagged text is thinking (preamble/CoT)
        // This prevents the "bubble flip" when the model starts talking before the <thought> tag.
        if (expectTags) {
          currentThought = text;
          currentAnswer = "";
        } else {
          currentAnswer = text;
          currentThought = "";
        }
      }
    }

    // Final global cleanup of any stray tags in the visible output
    answerText = _stripTags(currentAnswer);
    thoughtText = _stripTags(currentThought);
  }

  String _stripTags(String s) {
    return s.replaceAll('<thought>', '')
            .replaceAll('</thought>', '')
            .replaceAll('<final_answer>', '')
            .replaceAll('</final_answer>', '')
            .replaceAll('<turn|>', '')
            .replaceAll('<channel|>', '')
            .trim();
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'sessionId': sessionId,
      'role': role,
      'content': content,
      'createdAt': createdAt.toIso8601String(),
      'mediaPaths': mediaPaths?.join(','),
      'mediaType': mediaType,
    };
  }

  factory ChatMessage.fromMap(Map<String, dynamic> map) {
    return ChatMessage(
      id: map['id'],
      sessionId: map['sessionId'],
      role: map['role'],
      content: map['content'],
      createdAt: DateTime.parse(map['createdAt']),
      mediaPaths: map['mediaPaths'] != null ? (map['mediaPaths'] as String).split(',') : null,
      mediaType: map['mediaType'],
    );
  }
}
