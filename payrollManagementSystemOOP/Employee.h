#ifndef EMPLOYEE_H_INCLUDED
#define EMPLOYEE_H_INCLUDED

#include <ctime>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "Person.h"
#include "Position.h"
using namespace std;

class Employee: public Person {
    protected:
        int codeNumber;
        tm dateOfJoining;
        time_t currentTime;
        string department;
        Position* position;
        double startingSalary;

    public:
        class InvalidCodeNumber {
            private:
                int codeNumber;
            public:
                InvalidCodeNumber(int cN){
                    codeNumber = cN;
                }
                
                int getCodeNumber() const {
                    return codeNumber;
                }
        };

        class InvalidPhoneNumber {
            private:
                string number;
            public:
                InvalidPhoneNumber(const string& num){
                    number = num;
                }

                string getNumber() const {
                    return number;
                }
        };

        Employee(int cN, string n, string g, string pN, string dateJoin, string dep, Position* pos, double ss) {
                setName(n);
                setGender(g);
                setPhoneNumber(pN);
                codeNumber = cN;
                department = dep;
                position = pos;
                startingSalary = ss;
                
                istringstream iss(dateJoin);
                char separator;
                int year, month, day;
                iss >> year >> separator >> month >> separator >> day;

                tm tm = {};
                tm.tm_year = year - 1900;
                tm.tm_mon = month - 1;
                tm.tm_mday = day;
                dateOfJoining = tm;
            }

        int getCodeNumber() const{
            return codeNumber;
        }
        string getName() const{
            return name;
        }
        string getGender() const{
            return gender;
        }
        string getPhoneNumber() const{
            return phoneNumber;
        }
        string getDateOfJoining() const {
            ostringstream oss;
            oss << (1900 + dateOfJoining.tm_year) << "-"
                << (dateOfJoining.tm_mon + 1) << "-"
                << dateOfJoining.tm_mday;
            return oss.str();
        }
        string getDepartment() const{
            return department;
        }
        Position* getPosition() const{
            return position;
        }
        double getStartingSalary() const{
            return startingSalary;
        }

        virtual double calculateGrossSalary() = 0;  // Pure virtual function

        void display() const{
            cout << "Code number: " << codeNumber << endl;
            cout << "Name: " << name << endl;
            cout << "Gender: " << gender << endl;
            cout << "Phone: " << phoneNumber << endl;
            cout << "Start Date: " << getDateOfJoining() << endl;
            cout << "Department: " << department << endl;
            cout << "Position: " << position->getName() << endl;
        }

        void display(string info) const {  // Function overloading
            if (info == "salary_slip") {
                cout << "Code Number: " << codeNumber << endl;
                cout << "Name: " << name << endl;
                cout << "Position: " << position->getName() << endl;
                cout << "Department: " << department << endl;
            }
        }

        int getYearsOfService() const{
            time_t currentTime;  
            time(&currentTime);  
            tm* currentYear = localtime(&currentTime);  
            return currentYear->tm_year - dateOfJoining.tm_year;  
        }

        void setPhoneNumber(string newPhoneNumber){
            phoneNumber = newPhoneNumber;
        }
};

#endif // EMPLOYEE_H_INCLUDED