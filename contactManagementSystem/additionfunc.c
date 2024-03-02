#include <stdlib.h>
#include <stdio.h>
#include <string.h>
extern struct person *cPtr;
extern struct person contactP;
extern int slot;
extern struct person
{
  char fname[20];
  char lname[20];
  char phone[12];
};
int checkDupe(char former[], char latter[])
{
  if(strcmp(former, latter) ==  0)
  {
    return 1;
  }
  else
    return 0;
}

int checkNum(char num[])
{
    int len = strlen(num);
    for(int x = 0; x < len; x++) // use x < len, because EOF should be ignored here.
    {
        if (num[x] < 48 || num[x] > 57)
        {
            return 1;
        }

    }
    return 0;
}

void convertCase(char str[])
{
    int len = strlen(str);
  for (int i = 0; i <= len; i++)
  {
    if (str[i] >= 97 && str[i] <= 122)
    {
      str[i] -= 32;
    }
  }
}

void saveData()
{
    FILE *fpoint;
    fpoint = fopen("customer.dat", "wb");
    // write the data in binary to let the computer work faster(because raw binary is what computer understands)
    for(int i = 0; i < slot; i++)
    {
        fwrite((cPtr+i)->fname, sizeof(cPtr->fname), 1, fpoint);
        fwrite((cPtr+i)->lname, sizeof(cPtr->lname), 1, fpoint);
        fwrite((cPtr+i)->phone, sizeof(cPtr->phone), 1, fpoint);
    }
    fclose(fpoint);
}

void takeinput(char inputs[], int lbound, int rbound) //customized scanf to take string input with limited character to prevent buffer overflow.
{
  char input[40];
while(1)
    {

        scanf(" %[^\n]s", input); //input is basically an array in which array name = address name and will be kept outside this function.
        if (strlen(input)<lbound||strlen(input)>rbound)
            printf("Invalid! your string input should be %d-%d characters\n", lbound, rbound);
        else
        {
            strcpy(inputs, input);
            break;
        }
    }
}
