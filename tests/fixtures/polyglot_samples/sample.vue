<template><div class="point">{{ area(x) }}</div></template>
<script>
export default { data() { return { x: 1 }; }, methods: { area(n) { return n + 1; } } };
</script>
