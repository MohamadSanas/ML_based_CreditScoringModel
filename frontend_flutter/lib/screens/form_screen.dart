import 'package:flutter/material.dart';
import '../services/api_service.dart';

class FormScreen extends StatefulWidget {
  const FormScreen({super.key});

  @override
  State<FormScreen> createState() => _FormScreenState();
}

class _FormScreenState extends State<FormScreen> {
  // Controllers for numeric inputs
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

  // Helper widget for TextField
  Widget buildTextField(String label, TextEditingController controller) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(labelText: label),
        keyboardType: TextInputType.number,
      ),
    );
  }

  // Helper widget for Dropdown
  Widget buildDropdown(String label, String value, List<String> items, ValueChanged<String?> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: DropdownButtonFormField<String>(
        value: value,
        decoration: InputDecoration(labelText: label),
        items: items.map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
        onChanged: onChanged,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Loan Eligibility Form"),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: "About",
            onPressed: () {
              Navigator.pushNamed(context, '/about');
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Left Column
                Expanded(
                  child: Column(
                    children: [
                      buildTextField("Age", ageController),
                      buildTextField("Annual Income", incomeController),
                      buildTextField("Experience (yrs)", expController),
                      buildTextField("Credit Score", creditScoreController),
                      buildDropdown("Gender", selectedGender, ["Male", "Female", "Other"],
                          (val) => setState(() => selectedGender = val!)),
                      buildDropdown("Previous Loan Default", selectedPrevLoan, ["Yes", "No"],
                          (val) => setState(() => selectedPrevLoan = val!)),
                    ],
                  ),
                ),

                const SizedBox(width: 16),

                // Right Column
                Expanded(
                  child: Column(
                    children: [
                      buildTextField("Loan Amount", loanAmountController),
                      buildTextField("Credit History (yrs)", creditHistoryController),
                      buildTextField("Interest Rate", interestRateController),
                      buildDropdown("Education", selectedEducation, ["Associate", "Bachelor", "Doctorate", "High School", "Master"],
                          (val) => setState(() => selectedEducation = val!)),
                      buildDropdown("Home Ownership", selectedOwnership, ["MORTGAGE", "OTHER", "OWN", "RENT"],
                          (val) => setState(() => selectedOwnership = val!)),
                      buildDropdown("Loan Intent", selectedLoanIntent, ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
                          (val) => setState(() => selectedLoanIntent = val!)),
                    ],
                  ),
                ),
              ],
            ),

            const SizedBox(height: 24),

            // Predict Button centered
            isLoading
                ? const CircularProgressIndicator()
                : SizedBox(
                    width: 220,
                    child: ElevatedButton(
                      onPressed: _submit,
                      child: const Text("Predict Loan Eligibility"),
                    ),
                  ),
          ],
        ),
      ),
    );
  }
}
