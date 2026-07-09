<script lang="ts">
	import type { Mode } from '$lib/api';

	let {
		modes = [],
		value = $bindable<string>('balanced'),
		label = 'Mode'
	}: { modes: Mode[]; value?: string; label?: string } = $props();
</script>

<div class="space-y-2">
	<div class="text-[10px] uppercase tracking-[0.14em]" style="color: var(--ink-faint);">{label}</div>
	<div class="flex flex-wrap gap-1.5" role="radiogroup" aria-label={label}>
		{#each modes as m (m.name)}
			<button
				type="button"
				role="radio"
				aria-checked={value === m.name}
				onclick={() => (value = m.name)}
				class="px-3 py-1.5 rounded-md text-sm transition"
				style={value === m.name
					? 'background: var(--brand-dim); border: 1px solid rgba(201, 165, 84, 0.6); color: var(--brand); text-shadow: 0 0 8px rgba(201, 165, 84, 0.45);'
					: 'background: var(--surface); border: 1px solid var(--border); color: var(--ink-muted);'}
				title={m.description}
			>
				{m.name}
			</button>
		{/each}
	</div>
	{#if modes.length}
		<p class="text-xs text-balance" style="color: var(--ink-dim);">
			{modes.find((m) => m.name === value)?.description ?? ''}
		</p>
	{/if}
</div>
