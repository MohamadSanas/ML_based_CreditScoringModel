import 'package:flutter/material.dart';
import 'screens/form_screen.dart';
import 'screens/result_screen.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Loan Prediction App',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const HomeScreen(),
      routes: {
        '/form': (context) => const FormScreen(),   
        '/result': (context) => const ResultScreen(),
      },
    );
  }
}
