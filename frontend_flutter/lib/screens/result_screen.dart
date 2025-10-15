import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // Receive the result from FormScreen
    final result = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>? ?? {};

    final loanStatus = result['loan_status']?.toString() ?? "Unknown";
    final topFeatures = result['top_features'] as List<dynamic>? ?? [];

    return Scaffold(
      appBar: AppBar(title: const Text("Prediction Result")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Center(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                "Credit Score Result:",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 20),
              Text(
                loanStatus,
                style: const TextStyle(fontSize: 24, color: Colors.deepPurple),
              ),
              const SizedBox(height: 30),
              const Text(
                "Top Features Influencing Decision:",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 10),
              // Display SHAP top features
              ...topFeatures.map((feature) {
                return Text(
                  "${feature['Feature']}: ${feature['SHAP_value'].toStringAsFixed(3)}",
                  style: const TextStyle(fontSize: 16),
                );
              }).toList(),
            ],
          ),
        ),
      ),
    );
  }
}
