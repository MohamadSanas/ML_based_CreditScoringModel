import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';


class AboutPage extends StatelessWidget {
  const AboutPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "About Me",
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.deepPurple,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 👤 Profile Header
            Center(
              child: Column(
                children: [
                  const CircleAvatar(
                    radius: 50,
                    backgroundColor: Colors.deepPurple,
                    child: Icon(Icons.person, size: 60, color: Colors.white),
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    "Mohamad Sanas",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const Text(
                    "Computer Engineering Student, University of Jaffna",
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 30),

            // 🌟 Introduction Section
            const Text(
              "Hi, I’m Mohamad Sanas, a Computer Engineering student at the University of Jaffna with a passion for AI, full-stack development, and innovative problem-solving. I enjoy building intelligent systems that combine creativity with technology — from machine learning models to real-world applications that make a difference.",
              style: TextStyle(fontSize: 16, height: 1.5),
            ),

            const SizedBox(height: 20),

            

            const SizedBox(height: 30),

            // 📫 Contact Section
            const Text(
              "📫 Get in Touch",
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.deepPurple,
              ),
            ),
            const SizedBox(height: 10),
            const Text("Email: mohamadsanas23@gmail.com"),
            const Text("LinkedIn: linkedin.com/in/mohamad-sanas"),
          ],
        ),
      ),
    );
  }
}
