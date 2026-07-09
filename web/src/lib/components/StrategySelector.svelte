<script lang="ts">
	import type { Strategy } from '$lib/api';

	let {
		strategies = [],
		value = $bindable<string>('average')
	}: { strategies: Strategy[]; value?: string } = $props();
</script>

<div class="space-y-2">
	<div class="text-[10px] uppercase tracking-[0.14em]" style="color: var(--ink-faint);">
		Group strategy
	</div>
	<div class="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Group strategy">
		{#each strategies as s (s.name)}
			<button
				type="button"
				role="radio"
				aria-checked={value === s.name}
				onclick={() => (value = s.name)}
				class="px-3 py-1.5 rounded-md text-sm transition"
				style={value === s.name
					? 'background: var(--violet-dim); border: 1px solid rgba(143, 175, 122, 0.6); color: var(--green); text-shadow: 0 0 8px rgba(95, 135, 95, 0.5);'
					: 'background: var(--surface); border: 1px solid var(--border); color: var(--ink-muted);'}
				title={s.description}
			>
				{s.name}
			</button>
		{/each}
	</div>
	{#if strategies.length}
		<p class="text-xs text-balance" style="color: var(--ink-dim);">
			{strategies.find((s) => s.name === value)?.description ?? ''}
		</p>
	{/if}
</div>
