<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { Film, Users, User, BookOpen } from 'lucide-svelte';
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
		{ href: '/about', label: 'About', icon: BookOpen }
	];
</script>

<svelte:head>
	<title>movienight · group movie recommender</title>
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
	<link
		rel="stylesheet"
		href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600;1,700&display=swap"
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
					movie<span class="text-gradient">night</span>
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
				<a
					href="https://github.com/AlexW-F/letterboxdRecs"
					target="_blank"
					rel="noopener noreferrer"
					class="flex items-center gap-1.5 hover:opacity-100 transition"
					style="color: var(--ink-muted);"
					title="View source on GitHub"
					aria-label="GitHub repository"
				>
					<!-- Lucide-svelte dropped brand icons; inlining GitHub's official mark. -->
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="16"
						height="16"
						viewBox="0 0 24 24"
						fill="currentColor"
						aria-hidden="true"
					>
						<path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
					</svg>
				</a>
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
		class="border-t px-6 py-4 text-xs relative z-10"
		style="border-color: var(--border); color: var(--ink);"
	>
		<div class="max-w-6xl mx-auto flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-center">
			<span>movienight · group movie recommender · No tracking</span>
			<span style="color: var(--ink-dim);">·</span>
			<a href="/explore" class="underline hover:opacity-80" style="color: var(--ink);">
				Personalized 3D space
			</a>
			<span style="color: var(--ink-dim);">·</span>
			<a
				href="https://www.themoviedb.org/"
				target="_blank"
				rel="noopener noreferrer"
				class="inline-flex items-center gap-2 hover:opacity-80 transition"
				style="color: var(--ink);"
				title="Posters and streaming availability from TMDB"
			>
				<span>Posters &amp; streaming via</span>
				<img src="/tmdb-logo.svg" alt="TMDB" class="h-3 w-auto" />
			</a>
			<span class="hidden md:inline text-[10px]" style="color: var(--ink-muted);">
				— uses the TMDB API but is not endorsed or certified by TMDB.
			</span>
		</div>
	</footer>
</div>
