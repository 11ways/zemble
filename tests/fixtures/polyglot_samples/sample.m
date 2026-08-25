#import <Foundation/Foundation.h>
@protocol Shape
- (double)area;
@end
@interface Point : NSObject <Shape> { int x; }
@property int count;
- (instancetype)initWithX:(int)x;
+ (Point *)make;
@end
@implementation Point
- (double)area { return helper(x); }
+ (Point *)make { return [[Point alloc] initWithX:1]; }
- (void)run { [self area]; [Helper compute:1]; }
@end
static double helper(int n) { return n; }
