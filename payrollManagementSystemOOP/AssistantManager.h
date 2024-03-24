#ifndef ASSISTANTMANAGER_H_INCLUDED
#define ASSISTANTMANAGER_H_INCLUDED

#include "Employee.h"
#include "Position.h"
#include <cmath>

class AssistantManager: public Employee {
    public:
        AssistantManager(int cN, string n, string g, string pN, string dateJoin, string dep, Position* pos, double ss)
        : Employee(cN, n, g, pN, dateJoin, dep, pos, ss) {}

        double calculateGrossSalary() {
            return startingSalary * pow((1 + position->getAnnualIncrement()), getYearsOfService());
        }
};

#endif // ASSISTANTMANAGER_H_INCLUDED