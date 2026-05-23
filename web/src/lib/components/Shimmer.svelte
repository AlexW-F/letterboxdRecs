<script lang="ts">
	let {
		height = '1rem',
		width = '100%',
		rounded = '0.4rem'
	}: { height?: string; width?: string; rounded?: string } = $props();
</script>

<div class="shimmer-bg" style="height: {height}; width: {width}; border-radius: {rounded};"></div>
