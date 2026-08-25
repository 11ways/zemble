<script>
  export let x = 1;
  function area(n) { return helper(n); }
  function helper(n) { return n + 1; }
</script>
<div class="point">{area(x)}</div>
<style>.point { color: red; }</style>
