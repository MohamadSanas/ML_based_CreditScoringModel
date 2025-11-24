import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:url_launcher/url_launcher.dart';


class AboutPage extends StatelessWidget {
  const AboutPage({super.key});

  void _launchEmail() async {
    final Uri emailUri = Uri(
      scheme: 'mailto',
      path: 'mohamadsanas23@gmail.com',
      query: 'subject=Contact from Loan App',
    );

    if (!await launchUrl(
      emailUri,
      mode: LaunchMode.externalApplication, // Important for Web
    )) {
      debugPrint('Could not launch $emailUri');
    }
  }


  void _launchGitHub() async {
    final Uri githubUrl = Uri.parse('https://github.com/yourusername');
    if (!await launchUrl(
      githubUrl,
      mode: LaunchMode.externalApplication,
    )) {
      debugPrint('Could not launch $githubUrl');
    }
  }

  void _visitLinkedIn() async {
    final Uri linkedInUrl = Uri.parse('https://www.linkedin.com/in/mohamad-sanas-mohroof/');
    if (!await launchUrl(
      linkedInUrl,
      mode: LaunchMode.externalApplication,
    )) {
      debugPrint('Could not launch $linkedInUrl');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("About Me"),
        centerTitle: true,
        backgroundColor: const Color(0xFF008080),
        elevation: 4,
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Container(
              width: double.infinity,
              constraints: const BoxConstraints(
                maxWidth: 800,
                minHeight: 400,
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
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: const BoxDecoration(
                      color: Color(0xFF008080),
                      borderRadius: BorderRadius.only(
                        topLeft: Radius.circular(16),
                        topRight: Radius.circular(16),
                      ),
                    ),
                    
                    child: Column(
                      children: [
                        const Text(
                        "Mohroof Mohamad Sanas",
                        textAlign: TextAlign.left,
                          style: TextStyle(
                            fontSize: 18,
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),

                        const Text(
                        "Computer Engineering Student at University of Jaffna | Aspiring ML Developer",
                        textAlign: TextAlign.left,
                          style: TextStyle(
                            fontSize: 12,
                            color: Color.fromARGB(255, 255, 255, 255),
                            fontWeight: FontWeight.w400,
                          ),
                        ),
                      ],
                    ),

                    
                  ),
                  Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      children: [
                        CircleAvatar(
                          radius: 60,
                          backgroundImage:
                              AssetImage('assets/images/my_photo.jpg'),
                        ),
                        const SizedBox(height: 24),
                        const Text(
                          "Hello! I'm Mohroof Mohamad Sanas, a passionate developer with a keen interest in machine learning and its applications in finance. This Loan Eligibility Predictor app is one of my projects aimed at leveraging ML to assist individuals and lenders in making informed decisions about loan approvals. I hope you find this tool useful!",
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 16),
                        ),
                        const SizedBox(height: 24),
                        const Text(
                          "Feel free to reach out to me for collaborations or inquiries!",
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 14,
                            fontStyle: FontStyle.italic,
                          ),
                        ),

                        const SizedBox(height: 16),

                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            IconButton(
                              icon: const Icon(FontAwesomeIcons.github),
                              color: Colors.black,
                              iconSize: 32,
                              onPressed: _launchGitHub,
                              tooltip: 'GitHub Profile',
                            ),
                            const SizedBox(width: 24),
                            IconButton(
                              icon: const Icon(Icons.email),
                              color: const Color(0xFF008080),
                              iconSize: 32,
                              onPressed: _launchEmail,
                              tooltip: 'Send Email',
                            ),

                            const SizedBox(width: 24),

                            IconButton(
                              onPressed: _visitLinkedIn,
                              icon: const Icon(FontAwesomeIcons.linkedin),
                              color: Colors.blueAccent,
                              iconSize: 32,
                              tooltip: 'LinkedIn Profile',
                            )
                          ],
                        ),
                      ],
                    ),
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
