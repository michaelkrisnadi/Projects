#ifndef PERSON_H_INCLUDED
#define PERSON_H_INCLUDED

#include <string>
#include <iostream>
using namespace std;

class Person { 
    protected:
        string name;
        string gender;
        string phoneNumber;

    public:
        Person() = default;

        Person(string n, string g, string pN)
        {
            name = n;
            gender = g;
            phoneNumber = pN;
        }

        void setName(string n)
        {
            name = n;
        }

        void setGender(string g)
        {
            gender = g;
        }

        void setPhoneNumber(string pN)
        {
            phoneNumber = pN;
        }

        virtual void display() const = 0; 
};

#endif // PERSON_H_INCLUDED