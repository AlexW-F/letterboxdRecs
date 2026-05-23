<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { Film, Users, User, Sparkles, BookOpen } from 'lucide-svelte';
	import { getHealth, type Health } from '$lib/api';

	let { children } = $props();

	let health = $state<Health | null>(null);

	onMount(() => {
		const refresh = () =>
			getHealth()
				.then((h) => (health = h))
				.catch(() => (health = null));
		refresh();
		const id = setInterval(refresh, 30_000);
		return () => clearInterval(id);
	});

	const links = [
		{ href: '/', label: 'Group', icon: Users },
		{ href: '/me', label: 'Solo', icon: User },
		{ href: '/explore', label: 'Movie space', icon: Sparkles },
		{ href: '/about', label: 'About', icon: BookOpen }
	];
</script>

<svelte:head>
	<title>letterboxdRecs · cinephile recommender</title>
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
	<link
		rel="stylesheet"
		href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Instrument+Serif&display=swap"
	/>
</svelte:head>

<div class="min-h-screen flex flex-col grain relative">
	<header
		class="sticky top-0 z-30 border-b"
		style="
			background: rgba(10, 12, 16, 0.72);
			backdrop-filter: blur(14px) saturate(140%);
			-webkit-backdrop-filter: blur(14px) saturate(140%);
			border-color: var(--border);
		"
	>
		<div class="max-w-6xl mx-auto px-6 py-3 flex items-center gap-6">
			<a href="/" class="flex items-center gap-2 group">
				<span
					class="grid place-items-center w-8 h-8 rounded-md transition"
					style="background: linear-gradient(135deg, #34d399, #fbbf24); box-shadow: 0 4px 14px -4px rgba(52, 211, 153, 0.4);"
				>
					<Film size={16} color="#0a0c10" strokeWidth={2.5} />
				</span>
				<span class="font-semibold tracking-tight">
					letterboxd<span class="text-gradient">Recs</span>
				</span>
			</a>

			<nav class="hidden sm:flex items-center gap-1 text-sm">
				{#each links as l (l.href)}
					{@const active = page.url.pathname === l.href ||
						(l.href !== '/' && page.url.pathname.startsWith(l.href))}
					<a
						href={l.href}
						class="px-3 py-1.5 rounded-md transition flex items-center gap-1.5"
						style={active
							? 'background: var(--surface-strong); color: var(--ink); border: 1px solid var(--border-strong);'
							: 'color: var(--ink-muted); border: 1px solid transparent;'}
					>
						<l.icon size={14} strokeWidth={1.75} />
						{l.label}
					</a>
				{/each}
			</nav>

			<div class="ml-auto flex items-center gap-3 text-xs" style="color: var(--ink-dim);">
				{#if health?.status === 'ok'}
					<span class="flex items-center gap-1.5">
						<span
							class="inline-block w-1.5 h-1.5 rounded-full"
							style="background: #34d399; box-shadow: 0 0 8px #34d399;"
						></span>
						<span class="hidden md:inline">
							{health.catalog_size.toLocaleString()} films · SVD + ALS + genome
							{#if health.movie_space_loaded}· 3D viz{/if}
						</span>
						<span class="md:hidden">ok</span>
					</span>
				{:else}
					<span class="flex items-center gap-1.5">
						<span
							class="inline-block w-1.5 h-1.5 rounded-full animate-pulse"
							style="background: #f87171;"
						></span>
						<span>api offline</span>
					</span>
				{/if}
			</div>
		</div>
		<nav class="sm:hidden border-t px-4 py-1.5 flex gap-1 overflow-x-auto" style="border-color: var(--border);">
			{#each links as l (l.href)}
				{@const active = page.url.pathname === l.href ||
					(l.href !== '/' && page.url.pathname.startsWith(l.href))}
				<a
					href={l.href}
					class="px-3 py-1.5 rounded-md text-xs flex items-center gap-1 whitespace-nowrap"
					style={active
						? 'background: var(--surface-strong); color: var(--ink);'
						: 'color: var(--ink-muted);'}
				>
					<l.icon size={13} />
					{l.label}
				</a>
			{/each}
		</nav>
	</header>

	<main class="flex-1 relative z-10 px-4 sm:px-6 py-8 max-w-6xl w-full mx-auto">
		{@render children?.()}
	</main>

	<footer
		class="border-t px-6 py-4 text-xs text-center relative z-10"
		style="border-color: var(--border); color: var(--ink-faint);"
	>
		Personal cinephile recommender · No tracking · Stateless content-addressable cache
	</footer>
</div>
