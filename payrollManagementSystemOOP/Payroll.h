#ifndef PAYROLL_H_INCLUDED
#define PAYROLL_H_INCLUDED

#include <vector>
#include "Employee.h"
#include "Manager.h"
#include "JuniorExecutive.h"
#include "SeniorExecutive.h"
#include "AssistantManager.h"
#include "SeniorManager.h"
#include "Algorithm.h"
using namespace std;

class Payroll {
private:
    vector<Employee*> employees;

public:
    Payroll() {
    Position* juniorExec = new Position("Junior Executive", 0.06, 0.03, 100, 0, 0, 0);
    Position* seniorExec = new Position("Senior Executive", 0.07, 0.05, 150, 0, 0, 0);
    Position* asstManager = new Position("Assistant Manager", 0.08, 0.08, 250, 150, 0, 0);
    Position* manager = new Position("Manager", 0.10, 0.12, 350, 200, 0, 0.05);
    Position* seniorManager = new Position("Senior Manager", 0.12, 0.18, 400, 200, 800, 0.10);

    employees.push_back(new Manager(4128, "Jacky", "Male", "626-5860", "2013-05-02", "Account", manager, 7000));
    employees.push_back(new JuniorExecutive(1329, "Albert", "Male", "487-6331", "2022-04-01", "Account", juniorExec, 2800));
    employees.push_back(new AssistantManager(3477, "Lucie", "Female", "881-4560", "2020-06-18", "Customer Support", asstManager, 4700));
    employees.push_back(new SeniorManager(5511, "Harley", "Female", "700-2002", "2017-09-24", "Marketing", seniorManager, 9800));
    employees.push_back(new SeniorExecutive(2670, "Carmen", "Female", "672-4268", "2021-08-01", "Marketing", seniorExec, 4000));
    
    Algorithm::bubbleSort(employees);
    }

    void addEmployee(Employee* employee) {
        employees.push_back(employee);
    }

    void displayAllEmployees() {
        vector<Employee*> sortedEmployees = employees;
        Algorithm::bubbleSort(sortedEmployees);
        for (const auto& employee : sortedEmployees) {
            employee->display();
            cout << endl;
        }
        cout << "-----------------------------" << endl;
    }

    void displayEmployee(int codeNumber) {
        vector<Employee*> sortedEmployees = employees;
        Algorithm::bubbleSort(sortedEmployees);
        int index = Algorithm::binarySearch(sortedEmployees, 0, sortedEmployees.size() - 1, codeNumber); 
        if (index != -1) {
            cout << endl;
            sortedEmployees[index]->display();
            cout << "-----------------------------" << endl;
        }
        else{
            throw Employee::InvalidCodeNumber(codeNumber);
        }
    }

    void printSalarySlip(int code) {
        try {
            int idx = Algorithm::binarySearch(employees, 0, employees.size() - 1, code);

            if (idx != -1) {
                Employee* emp = employees[idx];  

                double grossSalary = emp->calculateGrossSalary();  

                double transportAllowance = emp->getPosition()->getTransportAllowance();
                double internetAllowance = emp->getPosition()->getInternetAllowance();
                double housingAllowance = emp->getPosition()->getHousingAllowance();
                double allowances = transportAllowance + internetAllowance + housingAllowance;
                double monthlyBonus = emp->getPosition()->getBonusRate() * grossSalary;
                double epf = (grossSalary + allowances) * 0.11;
                double tax = grossSalary * emp->getPosition()->getTaxRate();
                double netSalary = grossSalary + allowances + monthlyBonus - epf - tax;

                cout << endl;
                emp->display("salary_slip");
                cout << "-----------------------------" << endl;
                cout << "Current Gross Salary: $" << grossSalary << endl;
                if (transportAllowance > 0)
                    cout << "Transport Allowance: $" << transportAllowance << endl;
                if (internetAllowance > 0)
                    cout << "Internet Allowance: $" << internetAllowance << endl;
                if (housingAllowance > 0)
                    cout << "Housing Allowance: $" << housingAllowance << endl;
                cout << "Monthly Bonus: $" << monthlyBonus << endl;
                cout << "EPF Deduction: $" << epf << endl;
                cout << "Tax Deduction: $" << tax << endl;
                cout << "Current Net Income: $" << netSalary << endl;
                cout << "-----------------------------" << endl;

                return;
            }
            else {
                throw Employee::InvalidCodeNumber(code);
            }
        } catch(const Employee::InvalidCodeNumber& e) {
            throw;
        }
    }

    void updateContact(int code, const string& newContact) {
        for (char c: newContact) {
            if (!isdigit(c) && c != '-') {
                throw Employee::InvalidPhoneNumber(newContact);
            }
        }

        int start = 0;
        int end = employees.size() - 1;
        int idx = Algorithm::binarySearch(employees, start, end, code);

        if (idx != -1) {
            Employee* emp = employees[idx];
            emp->setPhoneNumber(newContact);
            cout << "Data Updated\n";
            cout << "-----------------------------" << endl;
            emp->display();
            cout << "-----------------------------" << endl;
        }
        else {
            throw Employee::InvalidCodeNumber(code);
        }
    }
};

#endif // PAYROLL_H_INCLUDED