import 'package:flutter/material.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("ML-Based Credit Scoring")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.credit_score, size: 80, color: Colors.deepPurple),
            SizedBox(height: 20),
            Text("Welcome to Credit Scoring App",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 40),
            ElevatedButton(
              onPressed: () => Navigator.pushNamed(context, '/form'),
              child: Text("Start Scoring"),
            ),
          ],
        ),
      ),
    );
  }
}
