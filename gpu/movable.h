#ifndef MOVABLE_H
#define MOVABLE_H

#include <iostream>
#include <optional>
#include <termios.h>
#include <unistd.h>

class movable {
public:
    virtual void move(char &) {}
};

inline std::optional<char> get_char(double dt_seconds)
{
    char val;
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    fd_set set;
    struct timeval timeout;
    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);
    timeout.tv_sec = static_cast<int>(dt_seconds);
    timeout.tv_usec = static_cast<int>((dt_seconds - timeout.tv_sec) * 1e6);

    int rv = select(STDIN_FILENO + 1, &set, NULL, NULL, &timeout);
    std::optional<char> result;
    if (rv > 0) {
        read(STDIN_FILENO, &val, 1);
        result = val;
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return result;
}

#endif // !MOVABLE_H
