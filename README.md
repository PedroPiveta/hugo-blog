# Hugo Blog

Este repositório contém um blog estático construído com Hugo.

## Visão geral

Projeto Hugo com configuração em `hugo.toml`, posts em `content/posts` e ativos estáticos em `static/`.

## Estrutura do projeto

- `archetypes/` : modelo de front matter para novos posts (`default.md`).
- `content/posts/` : arquivos Markdown dos posts do blog.
- `static/images/` : imagens públicas usadas nos posts (subpastas por post).
- `hugo.toml` : configuração do site Hugo.
- `build.sh` : script de build (uso em ambientes compatíveis com shell).
- `vercel.json` : configuração para deploy no Vercel.

## Pré-requisitos

- Hugo instalado (recomendado usar a versão compatível com seu tema).
- Git para clonar/gerenciar o repositório.

Nota: No Windows, use PowerShell, Git Bash ou WSL para executar scripts shell como `build.sh`.

## Rodando localmente (desenvolvimento)

1. Abra um terminal na raiz do projeto.
2. Rodar o servidor de desenvolvimento Hugo:

```powershell
hugo server -D
```

3. Acesse `http://localhost:1313` no navegador.

## Build para produção

Gerar os arquivos estáticos na pasta `public/`:

```powershell
hugo
```

Ou, se preferir usar o script (em ambiente compatível):

```bash
./build.sh
```

## Deploy

O projeto inclui `vercel.json` para deploy no Vercel. Para publicar no Vercel:

1. Conecte o repositório ao Vercel.
2. Configure o build command como `hugo` (ou use `npm`/outro se tiver pipeline).

Também é possível hospedar o diretório `public/` em qualquer serviço de arquivos estáticos.

## Como criar um novo post

1. Crie um arquivo Markdown em `content/posts/` com front matter (o `archetypes/default.md` contém um exemplo).
2. Coloque imagens em `static/images/<slug>/` e referencie-as no post com caminhos relativos a `/images/<slug>/`.

## Notas e dicas

- Mantenha imagens otimizadas para web em `static/images/`.
- Verifique `hugo.toml` para configurações do tema, baseURL e params do site.

## Ajuda

Abra uma issue neste repositório se encontrar problemas ou quiser melhorias.

---

Arquivo criado automaticamente com instruções básicas para desenvolvimento e deploy.
