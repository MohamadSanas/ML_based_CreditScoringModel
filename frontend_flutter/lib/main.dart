import 'package:flutter/material.dart';


import 'screens/home_screen.dart';
import 'screens/form_screen.dart';
import 'screens/result_screen.dart';
import 'screens/about_page.dart'; 



void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'ML-Based Credit Scoring App',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        scaffoldBackgroundColor: Colors.grey[100],
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.deepPurple,
          foregroundColor: Colors.white,
          centerTitle: true,
        ),
      ),

      // Initial screen
      home: const HomeScreen(),

      // Named routes
      routes: {
        '/form': (context) => const FormScreen(),
        '/result': (context) => const ResultScreen(),
        '/about': (context) => const AboutPage(), 
      },
    );
  }
}
