#include "Payroll.h"
#include <iostream>
#include <conio.h>
#include <limits>
using namespace std;

int main() {
    Payroll payroll;
    int choice;
    int codeNumber;
    string newContact;

    do {
        system("cls");
        cout << "Welcome to Payroll Management System" << endl;
        cout << "----------------------------------------" << endl;
        cout << "1 - Display All Employee Records" << endl;
        cout << "2 - Display Individual Employee Records" << endl;
        cout << "3 - Print Salary Slip" << endl;
        cout << "4 - Update Employee Contact" << endl;
        cout << "5 - End Program" << endl;
        cout << "----------------------------------------" << endl;
        
        while(true) {
            cout << "Enter your choice (1-5): ";
            cin >> choice;
            if (cin.fail()) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "ERROR: Invalid input. Please enter a number between 1 and 5." << endl;
            } else {
                break;
            }
        }

        try {
            switch(choice) {
                case 1:
                    payroll.displayAllEmployees();
                    cout << "Press any key to continue..." << endl;
                    getch();
                    system("cls");
                    break;

                case 2:
                    while(true) {
                        cout << "Enter employee code number: ";
                        cin >> codeNumber;
                        if (cin.fail()) {
                            cin.clear();
                            cin.ignore(numeric_limits<streamsize>::max(), '\n');
                            cout << "ERROR: Invalid input. Please enter a valid integer code number." << endl;
                        } 
                        else {
                            try {
                                payroll.displayEmployee(codeNumber);
                                break;
                            } catch(const Employee::InvalidCodeNumber& e) {
                                cout << "ERROR: Employee with code number " << e.getCodeNumber() << " not found." << endl;
                            }
                        }
                    }
                    cout << "Press any key to continue..." << endl;
                    getch();
                    system("cls");
                    break;

                case 3:
                    while(true) {
                        cout << "Enter employee code: ";
                        cin >> codeNumber;
                        if (cin.fail()) {
                            cin.clear();
                            cin.ignore(numeric_limits<streamsize>::max(), '\n');
                            cout << "ERROR: Invalid input. Please enter a valid integer code number." << endl;
                        } 
                        else {
                            try {
                                payroll.printSalarySlip(codeNumber);
                                break;
                            } catch(const Employee::InvalidCodeNumber& e) {
                                cout << "ERROR: Employee with code number " << e.getCodeNumber() << " not found." << endl;
                            }
                        }
                    }
                    cout << "Press any key to continue..." << endl;
                    getch();
                    system("cls");
                    break;

                case 4:
                    while (true) {
                        cout << "Enter employee code number: ";
                        cin >> codeNumber;
                        if (cin.fail()) {
                            cin.clear();
                            cin.ignore(numeric_limits<streamsize>::max(), '\n');
                            cout << "ERROR: Invalid input. Please enter a valid integer code number." << endl;
                        } 
                        else {
                            cout << "Enter new phone number: ";
                            cin >> newContact;
                            try {
                                payroll.updateContact(codeNumber, newContact);
                                break;
                            } catch(const Employee::InvalidCodeNumber& e) {
                                cout << "ERROR: Employee with code number " << e.getCodeNumber() << " not found." << endl;
                            } catch(const Employee::InvalidPhoneNumber& e) {
                                cout << "ERROR: " << e.getNumber() << " is an invalid phone number." << endl;
                            }
                        }
                    }
                    cout << "Press any key to continue..." << endl;
                    getch();
                    system("cls");
                    break;

                case 5:
                    cout << "Ending program..." << endl;
                    cout << "-----------------------------" << endl;
                    cout << "Press any key to finish" << endl;
                    getch();
                    system("cls");
                    break;

                default:
                    cout << "Invalid choice. Please select a valid option." << endl;
                    cout << "Press any key to continue..." << endl;
                    getch();
                    system("cls");
                    break;
            }
            
        } catch(const std::exception& e) {
            cout << "An unexpected error occured: " << e.what() << endl;
            cout << "Press any key to continue..." << endl;
            getch();
        }

    } while (choice != 5);

    return 0;
}
