<script lang="ts">
	let { name, size = 40 }: { name: string; size?: number } = $props();

	// Hash the name into a hue so each member gets a stable accent color.
	function hueFromName(s: string): number {
		let h = 0;
		for (let i = 0; i < s.length; i++) {
			h = (h * 31 + s.charCodeAt(i)) >>> 0;
		}
		return h % 360;
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
		background: linear-gradient(135deg, hsl({hue} 60% 55% / 0.85), hsl({(hue + 40) % 360} 60% 35% / 0.85));
		color: rgba(255, 255, 255, 0.95);
		box-shadow: 0 0 0 1px hsl({hue} 70% 70% / 0.25), inset 0 1px 0 hsl({hue} 90% 90% / 0.18);
		text-shadow: 0 1px 0 hsl({hue} 30% 20% / 0.4);
	"
>
	{initials || '?'}
</div>
