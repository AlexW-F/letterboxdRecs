<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';

	let { children } = $props();

	const links = [
		{ href: '/', label: 'Group' },
		{ href: '/me', label: 'Solo' },
		{ href: '/about', label: 'About' }
	];
</script>

<svelte:head>
	<title>letterboxdRecs</title>
</svelte:head>

<div class="min-h-screen flex flex-col">
	<header class="border-b border-white/10 px-6 py-4 flex items-center gap-6">
		<a href="/" class="font-semibold text-lg tracking-tight">letterboxdRecs</a>
		<nav class="flex gap-4 text-sm text-white/70">
			{#each links as l (l.href)}
				<a
					href={l.href}
					class="hover:text-white transition {page.url.pathname === l.href
						? 'text-white'
						: ''}"
				>
					{l.label}
				</a>
			{/each}
		</nav>
		<span class="ml-auto text-xs text-white/40">SVD + ALS + TF-IDF · ml-32m</span>
	</header>
	<main class="flex-1 px-6 py-8 max-w-6xl w-full mx-auto">
		{@render children?.()}
	</main>
	<footer class="border-t border-white/5 px-6 py-3 text-xs text-white/30 text-center">
		Personal cinephile recommender · No tracking · Stateless content-addressable cache
	</footer>
</div>
