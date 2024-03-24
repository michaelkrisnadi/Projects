#ifndef ALGORITHM_H_INCLUDED
#define ALGORITHM_H_INCLUDED

#include <vector>
#include "Employee.h"
using namespace std;

class Algorithm {
public:
    static void bubbleSort(vector<Employee*>& employees) {
        for (int i = 0; i < employees.size() - 1; i++) {
            for (int j = 0; j < employees.size() - i - 1; j++) {
                if (employees[j]->getCodeNumber() > employees[j + 1]->getCodeNumber()) {
                    swap(employees[j], employees[j + 1]);
                }
            }
        }
    }

    static int binarySearch(vector<Employee*>& employees, int start, int end, int codeNumber) {
        if (end >= start) {
            int mid = start + (end - start) / 2;
            if (employees[mid] -> getCodeNumber() == codeNumber){
                return mid;
            }
            if (employees[mid] -> getCodeNumber() > codeNumber){
                return binarySearch(employees, start, mid - 1, codeNumber);
            }
            return binarySearch(employees, mid + 1, end, codeNumber);
        }
        return -1;
    }
};

#endif // ALGORITHM_H_INCLUDED