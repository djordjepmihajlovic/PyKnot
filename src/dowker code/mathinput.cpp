#include <iostream>
#include <cmath>

int main(){

    // double x = 3.14;
    // double y = 6;
    // double z;
    // bool engineer = false;
    // z = std::max(x, y);
    // z = pow(x, y);
    // z = sqrt(x);
    // if (engineer == true){
    //     z = floor(x);
    // }
    // else{
    //     z = x;
    // }

    // if(z == 3){
    //     std::cout << "You're an charlatan " << z;
    // }
    // else{
    //     std::cout << "You're a genius " << z;
    // }

    // return 0;

    int grade = 59;
    bool happy = true;

    grade >= 60 ? std:: cout << "You pass!" << '\n': std::cout << "You fail!" << '\n';

    happy ? std:: cout << "Drinks?" << '\n' : std::cout << "Movie?" << '\n';

    if(grade<60 && happy){
        std::cout << "Mentality = Fuck it we ball" << '\n';
    }
    if(grade>=60 || happy){
        std::cout << "Mentality = mania";
    }
    return 0;
}