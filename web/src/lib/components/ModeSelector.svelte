<script lang="ts">
	import type { Mode } from '$lib/api';

	let {
		modes = [],
		value = $bindable<string>('balanced'),
		label = 'Mode'
	}: { modes: Mode[]; value?: string; label?: string } = $props();
</script>

<div class="space-y-2">
	<div class="text-xs uppercase tracking-wider text-white/40">{label}</div>
	<div class="flex flex-wrap gap-2">
		{#each modes as m (m.name)}
			<button
				type="button"
				onclick={() => (value = m.name)}
				class="px-3 py-1.5 rounded text-sm border transition
					{value === m.name
					? 'bg-emerald-500/20 border-emerald-400/70 text-emerald-100'
					: 'border-white/15 text-white/70 hover:border-white/30'}"
				title={m.description}
			>
				{m.name}
			</button>
		{/each}
	</div>
	{#if modes.length}
		<p class="text-xs text-white/40">
			{modes.find((m) => m.name === value)?.description ?? ''}
		</p>
	{/if}
</div>
