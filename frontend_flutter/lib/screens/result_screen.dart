import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final result = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>? ?? {};

    return Scaffold(
      appBar: AppBar(title: const Text("Prediction Result")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text("Credit Score Result:",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 20),
            Text(result['prediction']?.toString() ?? "Unknown",
                style: const TextStyle(fontSize: 24, color: Colors.deepPurple)),
            const SizedBox(height: 20),
            Text("Confidence: ${result['confidence']?.toString() ?? 'N/A'}%"),
          ],
        ),
      ),
    );
  }
}
