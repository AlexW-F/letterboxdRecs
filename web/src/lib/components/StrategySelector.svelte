<script lang="ts">
	import type { Strategy } from '$lib/api';

	let {
		strategies = [],
		value = $bindable<string>('average')
	}: { strategies: Strategy[]; value?: string } = $props();
</script>

<div class="space-y-2">
	<div class="text-xs uppercase tracking-wider text-white/40">Group strategy</div>
	<div class="flex flex-wrap gap-2">
		{#each strategies as s (s.name)}
			<button
				type="button"
				onclick={() => (value = s.name)}
				class="px-3 py-1.5 rounded text-sm border transition
					{value === s.name
					? 'bg-sky-500/20 border-sky-400/70 text-sky-100'
					: 'border-white/15 text-white/70 hover:border-white/30'}"
				title={s.description}
			>
				{s.name}
			</button>
		{/each}
	</div>
	{#if strategies.length}
		<p class="text-xs text-white/40">
			{strategies.find((s) => s.name === value)?.description ?? ''}
		</p>
	{/if}
</div>
