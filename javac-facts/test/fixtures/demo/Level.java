package demo;

/** An enum whose constants carry class bodies, so javac flattens them to Level$1 and Level$2. */
public enum Level {

    LOW {
        @Override
        public int weight() {
            return 1;
        }
    },
    HIGH {
        @Override
        public int weight() {
            return 10;
        }
    };

    public abstract int weight();
}
