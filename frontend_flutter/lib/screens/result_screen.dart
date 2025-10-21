import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final result =
        ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>? ?? {};

    final loanStatus = result['loan_status']?.toString() ?? "Unknown";

    final topFeatures = (result['top_features'] is List)
        ? result['top_features'] as List
        : [];

    return Scaffold(
      appBar: AppBar(
        title: const Text("Prediction Result"),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: "About",
            onPressed: () {
              Navigator.pushNamed(context, '/about'); // fixed route name
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
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
              ...topFeatures.map((feature) {
                final shapValue = feature['SHAP_value'];
                final shapText =
                    (shapValue is num) ? shapValue.toStringAsFixed(3) : shapValue.toString();
                return Text(
                  "${feature['Feature']}: $shapText",
                  style: const TextStyle(fontSize: 16),
                );
              }),
            ],
          ),
        ),
      ),
    );
  }
}
