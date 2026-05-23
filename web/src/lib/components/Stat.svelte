<script lang="ts">
	let {
		label,
		value,
		hint = ''
	}: { label: string; value: string | number; hint?: string } = $props();
</script>

<div class="surface px-4 py-3 anim-fade-up">
	<div
		class="text-[10px] uppercase tracking-[0.14em]"
		style="color: var(--ink-faint);"
	>
		{label}
	</div>
	<div class="text-2xl font-semibold mt-1 tabular-nums">
		{value}
	</div>
	{#if hint}
		<div class="text-xs mt-0.5" style="color: var(--ink-dim);">{hint}</div>
	{/if}
</div>
