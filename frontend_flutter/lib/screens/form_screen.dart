import 'package:flutter/material.dart';
import '../services/api_service.dart';

class FormScreen extends StatefulWidget {
  const FormScreen({super.key});

  @override
  State<FormScreen> createState() => _FormScreenState();
}

class _FormScreenState extends State<FormScreen> {
  // Text controllers for numeric inputs
  final TextEditingController ageController = TextEditingController();
  final TextEditingController incomeController = TextEditingController();
  final TextEditingController expController = TextEditingController();
  final TextEditingController creditScoreController = TextEditingController();
  final TextEditingController loanAmountController = TextEditingController();
  final TextEditingController creditHistoryController = TextEditingController();
  final TextEditingController interestRateController = TextEditingController();

  // Dropdown selections
  String selectedGender = "Male";
  String selectedPrevLoan = "No";
  String selectedEducation = "Bachelor";
  String selectedOwnership = "RENT";
  String selectedLoanIntent = "PERSONAL";

  bool isLoading = false;

  void _submit() async {
    setState(() => isLoading = true);

    final inputData = {
      "person_age": double.tryParse(ageController.text) ?? 0,
      "person_gender": selectedGender,
      "person_income": double.tryParse(incomeController.text) ?? 0,
      "person_emp_exp": double.tryParse(expController.text) ?? 0,
      "credit_score": double.tryParse(creditScoreController.text) ?? 0,
      "previous_loan_defaults_on_file": selectedPrevLoan,
      "education": selectedEducation,
      "home_ownership": selectedOwnership,
      "loan_intent": selectedLoanIntent,
      "loan_amnt": double.tryParse(loanAmountController.text) ?? 0,
      "credit_history_length": double.tryParse(creditHistoryController.text) ?? 0,
      "loan_interest_rate": double.tryParse(interestRateController.text) ?? 0,
    };

    try {
      final result = await ApiService.predictCreditScore(inputData);
      setState(() => isLoading = false);
      Navigator.pushNamed(context, '/result', arguments: result);
    } catch (e) {
      setState(() => isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Loan Eligibility Form")),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: ageController,
              decoration: const InputDecoration(labelText: "Age"),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: incomeController,
              decoration: const InputDecoration(labelText: "Annual Income"),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: expController,
              decoration: const InputDecoration(labelText: "Years of Experience"),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: creditScoreController,
              decoration: const InputDecoration(labelText: "Credit Score"),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: loanAmountController,
              decoration: const InputDecoration(labelText: "Loan Amount"),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: creditHistoryController,
              decoration: const InputDecoration(labelText: "Credit History Length"),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: interestRateController,
              decoration: const InputDecoration(labelText: "Loan Interest Rate"),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 20),

            // Gender dropdown
            DropdownButtonFormField<String>(
              initialValue: selectedGender,
              decoration: const InputDecoration(labelText: "Gender"),
              items: ["Male", "Female", "Other"]
                  .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                  .toList(),
              onChanged: (val) => setState(() => selectedGender = val!),
            ),

            // Previous loan
            DropdownButtonFormField<String>(
              initialValue: selectedPrevLoan,
              decoration: const InputDecoration(labelText: "Previous Loan Default"),
              items: ["Yes", "No"]
                  .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                  .toList(),
              onChanged: (val) => setState(() => selectedPrevLoan = val!),
            ),

            // Education
            DropdownButtonFormField<String>(
              initialValue: selectedEducation,
              decoration: const InputDecoration(labelText: "Education"),
              items: [
                "Associate",
                "Bachelor",
                "Doctorate",
                "High School",
                "Master"
              ].map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
              onChanged: (val) => setState(() => selectedEducation = val!),
            ),

            // Ownership
            DropdownButtonFormField<String>(
              initialValue: selectedOwnership,
              decoration: const InputDecoration(labelText: "Home Ownership"),
              items: ["MORTGAGE", "OTHER", "OWN", "RENT"]
                  .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                  .toList(),
              onChanged: (val) => setState(() => selectedOwnership = val!),
            ),

            // Loan Intent
            DropdownButtonFormField<String>(
              initialValue: selectedLoanIntent,
              decoration: const InputDecoration(labelText: "Loan Intent"),
              items: [
                "DEBTCONSOLIDATION",
                "EDUCATION",
                "HOMEIMPROVEMENT",
                "MEDICAL",
                "PERSONAL",
                "VENTURE"
              ].map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
              onChanged: (val) => setState(() => selectedLoanIntent = val!),
            ),

            const SizedBox(height: 25),
            isLoading
                ? const CircularProgressIndicator()
                : ElevatedButton(
                    onPressed: _submit,
                    child: const Text("Predict Loan Eligibility"),
                  ),
          ],
        ),
      ),
    );
  }
}
