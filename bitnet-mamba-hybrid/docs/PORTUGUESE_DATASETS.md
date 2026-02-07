# Datasets em Português do Brasil - Atualização

## Resumo das Mudanças

Este documento descreve a atualização dos datasets em português do Brasil no script de preprocessamento, adicionando alternativas de alta qualidade ao corpus OSCAR que está com acesso restrito.

## Novos Datasets Adicionados

### 1. **FineWeb2 Portuguese** (Prioridade 1)
- **Dataset ID**: `HuggingFaceFW/fineweb-2`
- **Configuração**: `pt` (português)
- **Tipo**: Corpus web multilíngue curado e de alta qualidade
- **Descrição**: Sucessor multilíngue do FineWeb-Edu, cobrindo mais de 1.000 idiomas com filtragem e curadoria similar ao fineweb-edu original
- **Qualidade**: Similar ao fineweb-edu (curado, filtrado, deduplificado)
- **Por que foi adicionado**: Equivalente em português do fineweb-edu usado para inglês

### 2. **Portuguese-PD (Public Domain)** (Prioridade 3)
- **Dataset ID**: `PleIAs/Portuguese-PD`
- **Tipo**: Maior corpus aberto de português
- **Descrição**: Agregação de todas as monografias e periódicos em português de domínio público
- **Qualidade**: Alta - conteúdo de domínio público, principalmente literário e acadêmico
- **Por que foi adicionado**: Maior corpus aberto de português disponível (março 2024)

### 3. **BrWaC (Brazilian Web as Corpus)** (Prioridade 5)
- **Dataset ID**: `UFRGS/brwac`
- **Tipo**: Corpus web brasileiro de grande escala
- **Estatísticas**: 3.53 milhões de documentos, 2.68 bilhões de tokens
- **Descrição**: Corpus web especificamente de português brasileiro
- **Qualidade**: Alta - corpus acadêmico, largamente usado em pesquisas de PLN
- **Por que foi adicionado**: Corpus brasileiro específico de grande escala

### 4. **Quati (Unicamp)** (Prioridade 7)
- **Dataset ID**: `unicamp-dl/quati`
- **Tipo**: Dataset acadêmico de português brasileiro
- **Descrição**: Dataset de alta qualidade desenvolvido pela Unicamp
- **Qualidade**: Alta - curado academicamente
- **Por que foi adicionado**: Dataset brasileiro acadêmico de qualidade

## Datasets Mantidos

Os seguintes datasets foram **mantidos** da versão anterior:

1. **Wikipedia Portuguese** (Prioridade 2) - `wikimedia/wikipedia` - Conteúdo enciclopédico de alta qualidade
2. **CulturaX Portuguese** (Prioridade 4) - `uonlp/CulturaX` - Corpus web multilíngue de alta qualidade
3. **OSCAR Portuguese** (Prioridade 6) - `oscar-corpus/OSCAR-2301` - Grande corpus web (pode ter acesso restrito)
4. **CC-100 Portuguese** (Prioridade 8) - `cc100` - Baseado em CommonCrawl
5. **MC4 Portuguese** (Prioridade 9) - `mc4` - Subconjunto português do C4 multilíngue
6. **BrWac Sample (legacy)** (Prioridade 10) - `eduagarcia/brwac` - Amostra do corpus brasileiro
7. **Portuguese News** (Prioridade 11) - `recogna-nlp/publico-news` - Artigos de notícias
8. **Carolina Corpus** (Prioridade 12) - `carolina-c4ai/corpus-carolina` - Corpus de referência brasileiro

## Sistema de Prioridades e Fallback

O script tenta carregar os datasets na ordem de prioridade. Se um dataset falhar (por acesso restrito, erro de rede, etc.), ele automaticamente tenta o próximo na lista.

**Ordem de tentativa**:
1. FineWeb2 Portuguese (curado, similar ao fineweb-edu)
2. Wikipedia Portuguese (sempre disponível, alta qualidade)
3. Portuguese-PD (maior corpus aberto)
4. CulturaX Portuguese (corpus web curado)
5. BrWaC oficial (corpus brasileiro de grande escala)
6. OSCAR Portuguese (pode ter restrições)
7. Quati (dataset acadêmico)
8. CC-100, mC4, e outros (fallback adicional)

## Características Principais

### FineWeb2 - Destaque Principal

O **FineWeb2** é o dataset mais importante adicionado, pois:
- É o equivalente multilíngue do fineweb-edu
- Usa pipeline de processamento similar ao fineweb original
- Cobre 1.000+ idiomas incluindo português
- Filtragem e curadoria de qualidade similar ao dataset inglês
- Publicado em 2025 como sucessor do FineWeb original

### Comparação com fineweb-edu

| Característica | fineweb-edu (Inglês) | FineWeb2 (Português) |
|----------------|----------------------|----------------------|
| Curadoria | ✓ | ✓ |
| Filtragem de qualidade | ✓ | ✓ |
| Deduplificação | ✓ | ✓ |
| Escala | 1.3T tokens | Multilíngue (PT subset) |
| Disponibilidade | Pública | Pública |

## Como Usar

### Preprocessamento Normal
```bash
# Usa o primeiro dataset disponível (FineWeb2)
python preprocess_datasets.py --output_dir ./data/tokenized
```

### Combinar Múltiplas Fontes
```bash
# Combina 3 fontes para maior diversidade
python preprocess_datasets.py --pt_combined_sources 3
```

### Processar Apenas Português
```bash
# Pula o inglês, processa só português
python preprocess_datasets.py --skip_english
```

## Requisitos de Acesso

Alguns datasets podem requerer:
- Autenticação no HuggingFace (token de acesso)
- Aceitar termos de uso
- Conexão estável com a internet

**Datasets que devem funcionar sem autenticação**:
- Wikipedia Portuguese
- CC-100 Portuguese
- MC4 Portuguese

**Datasets que podem precisar de autenticação**:
- FineWeb2 (verificar status no HuggingFace)
- Portuguese-PD (verificar status no HuggingFace)
- BrWaC (verificar status no HuggingFace)
- OSCAR Portuguese (sabidamente com restrições)

## Solução de Problemas

### Erro 403 Forbidden

Se você receber erro 403 ao carregar datasets:

1. **Faça login no HuggingFace**:
```bash
huggingface-cli login
```

2. **Aceite os termos de uso** da dataset no site do HuggingFace

3. **Use um token de acesso**:
```bash
export HF_TOKEN="seu_token_aqui"
```

### Dataset não disponível

O script automaticamente tentará o próximo dataset na lista de prioridades. Você verá mensagens de log indicando qual dataset está sendo tentado.

### Todos os datasets falharam

Se todos os datasets falharem:
1. Verifique sua conexão com a internet
2. Verifique se você está autenticado no HuggingFace
3. O script irá tentar Wikipedia como último fallback

## Referências

- [FineWeb2 Paper](https://arxiv.org/abs/2506.20920) - Pipeline multilíngue
- [Portuguese-PD](https://huggingface.co/datasets/PleIAs/Portuguese-PD) - Maior corpus PT
- [BrWaC](https://huggingface.co/datasets/UFRGS/brwac) - Corpus brasileiro
- [Quati](https://huggingface.co/datasets/unicamp-dl/quati) - Dataset Unicamp
- [Brazilian Portuguese Datasets Collection](https://huggingface.co/collections/ai-eldorado/brazilian-portuguese-datasets)

## Contribuindo

Se você conhece outros datasets de alta qualidade em português do Brasil, sinta-se à vontade para:
1. Adicioná-los à lista `PORTUGUESE_SOURCES` em `preprocess_datasets.py`
2. Definir a prioridade adequada
3. Testar com `test_portuguese_datasets.py`
4. Documentar aqui

## Atualização

**Data**: 2026-02-02
**Autor**: Atualização automática via Claude Code
**Motivo**: OSCAR Portuguese com acesso restrito
**Solução**: Adição de datasets alternativos de alta qualidade
