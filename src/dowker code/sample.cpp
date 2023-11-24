#include <iostream> 
#include <vector>
// iostream - header file for i/o operations

typedef std::string text_t; //two different ways of changing data type names -> typedef and using keywords (same thing AFAIK)
// using number_t = int;
typedef int number_t;

namespace first{ // namespace is a solution to name conflicts
    int x = 4; //decleration of x (whole integer)
}

namespace second{
    int x = 5;
}

namespace third{
    int x = 3;
}

int main(){
    using namespace third;

    // learning basic variables

    double pi; //decleration of unchangable float pi (number with decimal)

    std::cout << "What is the value of pi?";
    std::cin >> pi;
    char p = 'p';
    bool engineer = true;

    int correct = 9;
    int questions = 10;

    double score = correct/(double)questions * 100; // instance of type conversion 


    number_t o = 7;
    text_t pie = "pie"; // using type defs; shortening of data type code

    std::cout <<"Yo."<< '\n';
    // standard :: character output (<< this means output) "characters", '\n' is new line end with ;
    std::cout <<"Whats good?"<< '\n';
    std::cout << "I like " << x << '\n';
    std::cout << "Huh?" << '\n';
    std::cout << "I like " << pi << '\n';  
    std::cout << "What's wrong with you?" << '\n';
    std::cout << "I said I like " << pie << '\n';
    std::cout << "Oh... you're an engineer... " << '\n';
    std::cout << engineer << x << first::x << second::x << '\n';
    std::cout << "You've lost it man" << '\n';
    std::cout << score << "%";
    return 0; 
}

