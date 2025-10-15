import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

// Use relative import if package import fails
import 'package:supermarket_delivery_app/main.dart'; 

void main() {
  testWidgets('Loan Form UI test', (WidgetTester tester) async {
    // Build the app
    await tester.pumpWidget(const MyApp());

    // Check that all text fields are present
    expect(find.byType(TextField), findsNWidgets(7)); // 7 numeric fields
    expect(find.text('Age'), findsOneWidget);
    expect(find.text('Annual Income'), findsOneWidget);
    expect(find.text('Years of Experience'), findsOneWidget);
    expect(find.text('Credit Score'), findsOneWidget);
    expect(find.text('Loan Amount'), findsOneWidget);
    expect(find.text('Credit History Length'), findsOneWidget);
    expect(find.text('Loan Interest Rate'), findsOneWidget);

    // Check that all dropdowns are present
    expect(find.text('Gender'), findsOneWidget);
    expect(find.text('Previous Loan Default'), findsOneWidget);
    expect(find.text('Education'), findsOneWidget);
    expect(find.text('Home Ownership'), findsOneWidget);
    expect(find.text('Loan Intent'), findsOneWidget);

    // Interact with a dropdown
    await tester.tap(find.text('Male').first);
    await tester.pumpAndSettle();
    await tester.tap(find.text('Female').last);
    await tester.pumpAndSettle();

    // Enter values into numeric fields
    await tester.enterText(find.byType(TextField).at(0), '30'); // Age
    await tester.enterText(find.byType(TextField).at(1), '50000'); // Income
    await tester.enterText(find.byType(TextField).at(2), '5'); // Experience
    await tester.enterText(find.byType(TextField).at(3), '700'); // Credit Score
    await tester.enterText(find.byType(TextField).at(4), '15000'); // Loan Amount
    await tester.enterText(find.byType(TextField).at(5), '3'); // Credit History
    await tester.enterText(find.byType(TextField).at(6), '7'); // Interest Rate

    // Tap the submit button
    await tester.tap(find.byType(ElevatedButton));
    await tester.pumpAndSettle();

    // Since backend is not connected in tests, just verify button exists
    expect(find.text('Predict Loan Eligibility'), findsOneWidget);
  });
}
