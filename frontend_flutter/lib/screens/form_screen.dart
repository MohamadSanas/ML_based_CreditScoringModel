import 'package:flutter/material.dart';
import '../services/api_service.dart';

class FormScreen extends StatefulWidget {
  const FormScreen({super.key});

  @override
  State<FormScreen> createState() => _FormScreenState();
}

class _FormScreenState extends State<FormScreen> {
  // Controllers
  // ignore: non_constant_identifier_names
  final TextEditingController Dependent = TextEditingController();
  final TextEditingController incomeController = TextEditingController();
  final TextEditingController loanTerm = TextEditingController();
  final TextEditingController creditScoreController = TextEditingController();
  final TextEditingController loanAmountController = TextEditingController();
  final TextEditingController bankAssets = TextEditingController();
  final TextEditingController residentialAssets = TextEditingController();
  final TextEditingController commercialAssets = TextEditingController();
  // Dropdowns
  String selectedEducation = "Yes";
  String selectedEmployment = "Employed";
  


  bool isLoading = false;

  void _submit() async {
  setState(() => isLoading = true);

  final inputData = {
        "Dependent": double.tryParse(Dependent.text) ?? 0,
        "Education": selectedEducation.trim(), 
        "person_income": double.tryParse(incomeController.text) ?? 0,
        "Loan_term": double.tryParse(loanTerm.text) ?? 0,
        "credit_score": double.tryParse(creditScoreController.text) ?? 0,
        "Employment": selectedEmployment, 
        "loan_amnt": double.tryParse(loanAmountController.text) ?? 0,
        "bankAssets": double.tryParse(bankAssets.text) ?? 0,
        "residential_assets": double.tryParse(residentialAssets.text) ?? 0,
        "commercial_assets": double.tryParse(commercialAssets.text) ?? 0,
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
              constraints: const BoxConstraints(
                maxWidth: 800, 
                minHeight: 500
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
                                buildTextField("No of Dependent", Dependent),
                                buildTextField("Annual Income", incomeController),
                                buildTextField("Loan Term", loanTerm),
                                buildTextField("Credit Score", creditScoreController),
                                buildTextField("Commercial Assets", commercialAssets),
                                buildDropdown("Graduated ?", selectedEducation, ["Yes", "No"],
                                    (val) => setState(() => selectedEducation = val!)),
                                buildDropdown(
                                    "Employment status", selectedEmployment, ["Employed", "Unemployed"],
                                    (val) => setState(() => selectedEmployment = val!)),
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
                                buildTextField("Bank Assets", bankAssets),
                                buildTextField("Residential Assets", residentialAssets),                              
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
