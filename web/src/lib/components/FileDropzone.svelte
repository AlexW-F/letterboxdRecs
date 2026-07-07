<script lang="ts">
	import { Upload, CheckCircle2 } from 'lucide-svelte';
	import { fly } from 'svelte/transition';

	let {
		label,
		accept = '.csv',
		file = $bindable<File | null>(null),
		hint = ''
	}: {
		label: string;
		accept?: string;
		file?: File | null;
		hint?: string;
	} = $props();

	let dragging = $state(false);

	function onFile(f: File | null | undefined) {
		if (!f) return;
		file = f;
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragging = false;
		onFile(e.dataTransfer?.files?.[0]);
	}

	function onChange(e: Event) {
		const input = e.target as HTMLInputElement;
		onFile(input.files?.[0]);
	}
</script>

<label
	class="dropzone block rounded-lg border-2 border-dashed cursor-pointer transition px-4 py-5 text-center relative overflow-hidden"
	style={file
		? 'background: var(--brand-dim); border-color: rgba(52,211,153,0.55);'
		: dragging
			? 'background: rgba(52,211,153,0.06); border-color: var(--brand);'
			: 'background: rgba(0,0,0,0.2); border-color: var(--border);'}
	ondragover={(e) => {
		e.preventDefault();
		dragging = true;
	}}
	ondragleave={() => (dragging = false)}
	ondrop={onDrop}
>
	<div class="flex items-center justify-center gap-2 mb-1">
		{#if file}
			<CheckCircle2 size={16} style="color: var(--brand);" />
		{:else}
			<Upload size={16} style="color: var(--ink-dim);" />
		{/if}
		<div class="text-sm font-medium">{label}</div>
	</div>
	{#if file}
		<div in:fly={{ y: -2, duration: 180 }} class="text-xs mono" style="color: #6ee7b7;">
			{file.name}
			<span style="color: var(--ink-dim);">· {(file.size / 1024).toFixed(0)} KB</span>
		</div>
	{:else}
		<div class="text-xs" style="color: var(--ink-faint);">
			{hint || 'drop a CSV or click to choose'}
		</div>
	{/if}
	<!-- sr-only (not display:none) keeps the input tabbable, so keyboard
	     users can reach the picker; the ring renders on the label. -->
	<input type="file" {accept} class="sr-only" onchange={onChange} />
</label>

<style>
	.dropzone:focus-within {
		outline: 2px solid rgba(52, 211, 153, 0.7);
		outline-offset: 2px;
	}
</style>
