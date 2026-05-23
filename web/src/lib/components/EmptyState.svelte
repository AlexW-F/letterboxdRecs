<script lang="ts">
	import type { Component, Snippet } from 'svelte';

	let {
		icon,
		title,
		description = '',
		children
	}: {
		icon?: Component;
		title: string;
		description?: string;
		children?: Snippet;
	} = $props();
</script>

<div class="surface p-8 text-center space-y-3 anim-fade-up">
	{#if icon}
		<div class="grid place-items-center w-12 h-12 mx-auto rounded-full chip-violet">
			{@const Icon = icon}
			<Icon size={22} strokeWidth={1.5} />
		</div>
	{/if}
	<h3 class="text-lg font-medium">{title}</h3>
	{#if description}
		<p class="text-sm text-balance" style="color: var(--ink-muted); max-width: 36ch; margin: 0 auto;">
			{description}
		</p>
	{/if}
	{#if children}
		<div class="pt-1 flex justify-center">{@render children()}</div>
	{/if}
</div>
