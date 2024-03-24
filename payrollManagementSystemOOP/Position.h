#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <string>
#include <cmath> 
using namespace std;

class Position {
    private:
        string name;
        double annualIncrement;
        double taxRate;
        double transportAllowance;
        double internetAllowance;
        double housingAllowance;
        double monthlyBonusRate;

    public:
        Position(string n, double ai, double tr, double ta, double ia, double ha, double mbr)
        {
            name = n;
            annualIncrement = ai;
            taxRate = tr;
            transportAllowance = ta;
            internetAllowance = ia;
            housingAllowance = ha;
            monthlyBonusRate = mbr;
        }

        string getName(){
            return name;
        }
        double getAnnualIncrement(){
            return annualIncrement;
        }
        double getTaxRate(){
            return taxRate;
        }
        double getTransportAllowance(){
            return transportAllowance;
        }
        double getInternetAllowance(){
            return internetAllowance;
        }
        double getHousingAllowance(){
            return housingAllowance;
        }
        double getBonusRate(){
            return monthlyBonusRate;
        }

        double getTotalAllowances(){
            return transportAllowance + internetAllowance + housingAllowance;
        }

        double calculateGrossSalary(double startSalary, int years) const{
            double grossSalary = startSalary;
            for (int i = 0; i < years; ++i) {
                grossSalary += grossSalary * (annualIncrement / 100.0);
            }
            return grossSalary;
        }
};

#endif // POSITION_H_INCLUDED