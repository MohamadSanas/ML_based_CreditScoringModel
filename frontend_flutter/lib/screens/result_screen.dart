import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // Retrieve result from arguments, if passed
    final result =ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>? ?? {};

    final loanStatus = result['loan_status']?.toString() ?? "Unknown";

    final topFeatures = (result['top_features'] is List)
        ? result['top_features'] as List
        : [];


    return Scaffold(
      appBar: AppBar(
        title: const Text("Result"),
        centerTitle: true,
        backgroundColor: const Color(0xFF008080),
        elevation: 4,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: "About",
            onPressed: () => Navigator.pushNamed(context, '/about'),
          ),
        ],
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Container(
              width: double.infinity,
              constraints: const BoxConstraints(
                maxWidth: 600, // like max-w-4xl
                minHeight: 200,
              ),


              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black26,
                    spreadRadius: 4,
                    blurRadius: 12,
                    offset: Offset(0, 4),
                  ),
                ],

              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Header
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: const BoxDecoration(
                      color: Color(0xFF008080),
                      borderRadius: BorderRadius.only(
                        topLeft: Radius.circular(16),
                        topRight: Radius.circular(16),
                      ),
                    ),


                    child: const Text(
                      "Loan Eligibility Result",
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),

                  // Result content
                  Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      children: [
                        const Text(
                        "Credit Score Result:",
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold
                            ),
                        ),
                        const SizedBox(height: 15),

                        Text(
                          loanStatus,
                          style: const TextStyle(
                            fontSize: 24, 
                            color: Color.fromARGB(255, 0, 0, 0)
                          ),
                        ),
                        const SizedBox(height: 15),

                        ElevatedButton(
                          onPressed: () {
                            Navigator.pushNamedAndRemoveUntil(
                              context, 
                              '/form', 
                              (route) => false
                            );
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFFE69900),
                            padding: const EdgeInsets.symmetric(
                              horizontal: 24, 
                              vertical: 12
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                          child: const Text("Make Another Prediction",
                          style: TextStyle(color: Color.fromARGB(255, 0, 0, 0)),),

                        ),
                      ]
                    )
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
