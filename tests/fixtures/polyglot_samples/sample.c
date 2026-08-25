#include <stdio.h>
#define MAX 3
struct point { int x; int y; };
typedef struct { int a; } thing_t;
enum color { RED, GREEN };
union u { int a; float b; };
static int count;
int helper(int n);
int helper(int n) { return n + 1; }
static void run(struct point *p, int n) { printf("%d", helper(n)); p->x = n; }
int main(void) { run(NULL, 1); return 0; }
