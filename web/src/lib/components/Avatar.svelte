<script lang="ts">
	let { name, size = 40 }: { name: string; size?: number } = $props();

	// Hash the name into a stable hue, constrained to the warm miasma band
	// (rust 18° → moss 112°) so members stay distinct but on-palette.
	function hueFromName(s: string): number {
		let h = 0;
		for (let i = 0; i < s.length; i++) {
			h = (h * 31 + s.charCodeAt(i)) >>> 0;
		}
		return 18 + (h % 95);
	}

	const hue = $derived(hueFromName(name));
	const initials = $derived(
		name
			.split(/\s+/)
			.map((p) => p[0])
			.filter(Boolean)
			.slice(0, 2)
			.join('')
			.toUpperCase()
	);
</script>

<div
	class="grid place-items-center rounded-full font-semibold flex-shrink-0"
	style="
		width: {size}px;
		height: {size}px;
		font-size: {Math.max(11, size * 0.42)}px;
		background: linear-gradient(135deg, hsl({hue} 42% 52% / 0.9), hsl({hue + 22} 40% 34% / 0.9));
		color: rgba(27, 26, 23, 0.92);
		box-shadow: 0 0 0 1px hsl({hue} 50% 68% / 0.3), inset 0 1px 0 hsl({hue} 60% 85% / 0.2);
		text-shadow: 0 1px 0 hsl({hue} 35% 72% / 0.35);
	"
>
	{initials || '?'}
</div>
