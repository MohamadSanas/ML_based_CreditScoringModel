import 'package:flutter/material.dart';
import '../services/api_service.dart';

class FormScreen extends StatefulWidget {
  const FormScreen({super.key});

  @override
  State<FormScreen> createState() => _FormScreenState();
}

class _FormScreenState extends State<FormScreen> {
  // Controllers
  final TextEditingController ageController = TextEditingController();
  final TextEditingController incomeController = TextEditingController();
  final TextEditingController expController = TextEditingController();
  final TextEditingController creditScoreController = TextEditingController();
  final TextEditingController loanAmountController = TextEditingController();
  final TextEditingController creditHistoryController = TextEditingController();
  final TextEditingController interestRateController = TextEditingController();

  // Dropdowns
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
        "person_gender": selectedGender.trim(), // send exact string
        "person_income": double.tryParse(incomeController.text) ?? 0,
        "person_emp_exp": double.tryParse(expController.text) ?? 0,
        "credit_score": double.tryParse(creditScoreController.text) ?? 0,
        "previous_loan_defaults_on_file": selectedPrevLoan, // "Yes" or "No"
        "education": selectedEducation.trim(),
        "home_ownership": selectedOwnership.trim(),
        "loan_intent": selectedLoanIntent.trim(),
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


  // TextField builder
  Widget buildTextField(String label, TextEditingController controller) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(
          labelText: label,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        ),
        keyboardType: TextInputType.number,
      ),
    );
  }

  // Dropdown builder
  Widget buildDropdown(
      String label, String value, List<String> items, ValueChanged<String?> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: DropdownButtonFormField<String>(
        initialValue: value,
        decoration: InputDecoration(
          labelText: label,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        ),
        items: items.map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
        onChanged: onChanged,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FB),
      appBar: AppBar(
        title: const Text("Loan Application Form"),
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
              constraints: const BoxConstraints(maxWidth: 800, minHeight: 500),
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
                      "Loan Eligibility Form",
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 24,
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),

                  const SizedBox(height: 20),

                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal, // enable horizontal scrolling
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Left Column
                          SizedBox(
                            width: 300, // min width for left column
                            child: Column(
                              children: [
                                buildTextField("Age", ageController),
                                buildTextField("Monthly Income", incomeController),
                                buildTextField("Experience (yrs)", expController),
                                buildTextField("Credit Score", creditScoreController),
                                buildDropdown("Gender", selectedGender, ["Male", "Female", "Other"],
                                    (val) => setState(() => selectedGender = val!)),
                                buildDropdown(
                                    "Previous Loan Default", selectedPrevLoan, ["Yes", "No"],
                                    (val) => setState(() => selectedPrevLoan = val!)),
                              ],
                            ),
                          ),
                          const SizedBox(width: 16),
                          // Right Column
                          SizedBox(
                            width: 300, // min width for right column
                            child: Column(
                              children: [
                                buildTextField("Loan Amount", loanAmountController),
                                buildTextField("Credit History (yrs)", creditHistoryController),
                                buildTextField("Interest Rate", interestRateController),
                                buildDropdown("Education", selectedEducation,
                                    ["Associate", "Bachelor", "Doctorate", "High School", "Master"],
                                    (val) => setState(() => selectedEducation = val!)),
                                buildDropdown("Home Ownership", selectedOwnership,
                                    ["MORTGAGE", "OTHER", "OWN", "RENT"],
                                    (val) => setState(() => selectedOwnership = val!)),
                                buildDropdown(
                                    "Loan Intent",
                                    selectedLoanIntent,
                                    [
                                      "DEBTCONSOLIDATION",
                                      "EDUCATION",
                                      "HOMEIMPROVEMENT",
                                      "MEDICAL",
                                      "PERSONAL",
                                      "VENTURE"
                                    ],
                                    (val) => setState(() => selectedLoanIntent = val!)),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 24),

                  // Button + Loading
                  Center(
                    child: isLoading
                        ? const CircularProgressIndicator(color: Color(0xFF008080))
                        : SizedBox(
                            width: 220,
                            child: ElevatedButton(
                              onPressed: _submit,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFFE69900),
                                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(8)),
                              ),
                              child: const Text("Predict Loan Eligibility"),
                            ),
                          ),
                  ),
                  const SizedBox(height: 16),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
