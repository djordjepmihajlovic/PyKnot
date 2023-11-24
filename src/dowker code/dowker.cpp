#include <iostream>
#include <fstream>
#include <string>

int main(){
    
    std::fstream newfile;
    newfile.open("/Users/djordjemihajlovic/Desktop/knot_1.txt", std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        while(std::getline(newfile, tp)){
            std::cout << tp <<'\n';
        }
        newfile.close();
    }

}