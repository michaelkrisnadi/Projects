#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <conio.h>
#define N 100
struct person{
  char fname[20];
  char lname[20];
  char phone[12];
}contactP[N];
struct person *cPtr = contactP; // declared pointers for structures.
int slot = 0; // how many slot occupied.
extern void convertCase(char str[]); // function prototype for converting each alphabetical array of character to uppercase.
void addContact(), deleteContact(), viewContact(), editContact();
extern void saveData();
extern void takeinput(char inputs[], int lbound, int rbound); // function prototype.
int menu();
extern int checkDupe(char former[], char latter[]); // check whether the same data has already existed or not.
extern int checkNum(char num[]); // check whether the inputted strings are numbers or not.
int main()
{
FILE *fcustomer;
fcustomer = fopen("customer.dat", "rb");
while(fread(cPtr+slot, sizeof(struct person), 1, fcustomer)==1)
    slot++;
fclose(fcustomer);
saveData();
while(1)
switch (menu())
{
case '1':
    if (slot > N)
    {
       system("cls");
       printf("Data exceeds the limit [%d]\n", N);
       printf("Type any key to go back to menu");
       getch();
       continue;
    }

    addContact();
    continue;
case '2':
    editContact();
    continue;
case '3':
    if (slot == 0)
    {
        system("cls");
        printf("There's no data to be deleted!\n");
        printf("Type any key to go back to menu");
        getch();
        continue;
    }
    deleteContact();
    continue;
case '4':
    viewContact();
    printf("Press any key to return to menu.");
    getch();
    continue;
case '5':
    system("cls");
    saveData();
    printf("Data has been successfully saved\n");
    printf("Press any key to return to menu.");
    getch();
    continue;
case 'X':
case 'x':
    saveData(); // so basically, the data will be saved properly only when we exit properly
    exit(1);

default:
    continue;

}


}
int menu()
{
    char choice[3]="ch";
    while(strlen(choice)!= 1)
    {
    system("cls");
    // printing choice
    puts("\tContact Management System  ");
    puts("\tSelect a number to proceed! \n");
    puts("\t1. Add");
    puts("\t2. Edit");
    puts("\t3. Delete");
    puts("\t4. View");
    puts("\t5. Save Data");
    puts("\t\nEnter X to quit (automatically save data)");
    scanf(" %s", choice);
    }
    return choice[0];
}
void addContact()
{
system("cls");
while(1)
    {
    char fname[20], lname[20], phone[12];
    printf("First name: ");
    takeinput(fname, 0, 19);
    printf("Last name: ");
    takeinput(lname, 0, 19);
    convertCase(fname); // convert every array of characters to uppercase
    convertCase(lname);
    printf("Phone number: (10-11 digit numbers)\n");
    while(1)
    {
    takeinput(phone, 10, 11);
    if (checkNum(phone))
        printf("There's a reason why it is called phone NUMBERS!\n");
    else
        break;
    }
    int check1 = 0, check2 = 0, check3 = 0; // set counter variable on check1 for firstname, check2 for lastname, check3 for phonenumber.
    for(int count = 0; count < slot; count++)
    {
        check1 += checkDupe(fname, (cPtr+count)->fname);
        check2 += checkDupe(lname, (cPtr+count)->lname);
        check3 += checkDupe(phone, (cPtr+count)->phone);
    }

    if (check1+check2+check3 == 3)
        {
            char YN;
            printf("Woops! duplicate data is not a wise choice.\n");
            printf("[1] Back to menu? [2] Try again\n");
            scanf(" %c", &YN);
            if (YN == '1')
                break; // get out loop while(1)
            else if (YN == '2')
                continue; // reloop while(1)
            else
                break;
        }
    strcpy((cPtr+slot)->fname, fname);
    strcpy((cPtr+slot)->lname, lname);
    strcpy((cPtr+slot)->phone, phone);
    slot++;
    break;
    }

}
void deleteContact()
{
system("cls");
char fname[20], lname[20], phone[12];
    printf("First name: ");
    takeinput(fname, 0, 19);
    printf("Last name: ");
    takeinput(lname, 0, 19);
    convertCase(fname); // convert every array of characters to uppercase
    convertCase(lname);
    printf("Phone number: (10-11 digit numbers)\n");
    while(1)
    {
    takeinput(phone, 10, 11);
    if (checkNum(phone))
        printf("There's a reason why it is called phone NUMBERS!\n");
    else
        break;
    }
for(int i = 0; i < slot; i++)
    if(checkDupe(fname, (cPtr+i)->fname)) // succession nested if, (cPtr+i) as available structure, all info refers to (cPtr+i) should match.
        if(checkDupe(lname, (cPtr+i)->lname))
            if(checkDupe(phone, (cPtr+i)->phone))
            {
                slot--;
                printf("Contact has been deleted.\n[Press any key to continue]");
                getch();
                for(; i < slot; i++)
                    *(cPtr+i) = *(cPtr+i+1);// the previous (deleted) one get replaced by next block array of structure.
            }
}
void editContact()
    {
system("cls");
char fname[20], lname[20], phone[12];
    printf("First name: ");
    takeinput(fname, 0, 19);
    printf("Last name: ");
    takeinput(lname, 0, 19);
    convertCase(fname); // convert every array of characters to uppercase
    convertCase(lname);
    printf("Phone number: (10-11 digit numbers)\n");
    while(1)
    {
    takeinput(phone, 10, 11);
    if (checkNum(phone))
        printf("There's a reason why it is called phone NUMBERS!\n");
    else
        break;
    }

int exist = 0; // determine whether the data exist or not
for(int i = 0; i < slot; i++)
    if(checkDupe(fname, (cPtr+i)->fname)) // succession nested if, (cPtr+i) as available structure, all info refers to (cPtr+i) should match.
        if(checkDupe(lname, (cPtr+i)->lname))
            if(checkDupe(phone, (cPtr+i)->phone))
            { //basically the whole add function printed here with slot changed to i. (destination is cPtr+i)
            while(1)
            {
                printf("First name: ");
                takeinput(fname, 0, 19);
                printf("Last name: ");
                takeinput(lname, 0, 19);
                convertCase(fname); // convert every array of characters to uppercase
                convertCase(lname);
                printf("Phone number: (10-11 digit numbers)\n");
                while(1)
                {
                takeinput(phone, 10, 11);
                if (checkNum(phone))
                    printf("There's a reason why it is called phone NUMBERS!\n");
                else
                    break;
                }
                int check1 = 0, check2 = 0, check3 = 0; // set counter variable on check1 for firstname, check2 for lastname, check3 for phonenumber.
                for(int count = 0; count < slot; count++)
                    {
                    check1 += checkDupe(fname, (cPtr+count)->fname);
                    check2 += checkDupe(lname, (cPtr+count)->lname);
                    check3 += checkDupe(phone, (cPtr+count)->phone);
                    }

                if (check1+check2+check3 == 3) // so, one person can have multiple contacts, or one contacts could also be owned by multiple persons (maybe one person as the host of another person)
                    {
                        char YN;
                        printf("Either the data is already available or inputs are equivalent as prior data.\n");
                        printf("Confused? view the contact list!\n");
                        printf("[1] Back to menu? [2] Try again\n");
                        scanf(" %c", &YN);
                        if (YN == '1')
                            break; // get out loop while(1)
                        else if (YN == '2')
                            continue; // reloop while(1)
                        else
                            break;
                    }
                strcpy((cPtr+i)->fname, fname);
                strcpy((cPtr+i)->lname, lname);
                strcpy((cPtr+i)->phone, phone);
                exist = 1;
                break;
            }
            }
if (exist == 1)
{
    printf("Data has been edited\n");
    printf("Press any key to proceed\n");
    getch();
}
else
{
    printf("No such data, back to menu\n");
    printf("Press any key to proceed\n");
    getch();
}


}

void viewContact()
{
system("cls");
printf("\t*** View Contact Lists ***\n\n\n");
for(int i = 0; i < slot; i++)
{
    printf("\t<Contact %d>\n\n", i+1);
    printf("\tFirst name: %s\n", (cPtr+i)->fname);
    printf("\tLast name: %s\n", (cPtr+i)->lname);
    printf("\tPhone number: %s\n\n", (cPtr+i)->phone);
}
}


