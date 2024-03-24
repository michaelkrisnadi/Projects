#ifndef JUNIOREXECUTIVE_H_INCLUDED
#define JUNIOREXECUTIVE_H_INCLUDED

#include "Employee.h"
#include "Position.h"
#include <cmath>

class JuniorExecutive: public Employee {
    public:
        JuniorExecutive(int cN, string n, string g, string pN, string dateJoin, string dep, Position* pos, double ss)
        : Employee(cN, n, g, pN, dateJoin, dep, pos, ss) {}

        double calculateGrossSalary() {
            return startingSalary * pow((1 + position->getAnnualIncrement()), getYearsOfService());
        }
};

#endif // JUNIOREXECUTIVE_H_INCLUDED