import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'http://127.0.0.1:5000'; // Flask backend URL

  static Future<Map<String, dynamic>> predictCreditScore(
      Map<String, dynamic> inputData) async {
    final url = Uri.parse('$baseUrl/predict'); // Flask route e.g. /predict

    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(inputData),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to predict. Status: ${response.statusCode}');
    }
  }
}
