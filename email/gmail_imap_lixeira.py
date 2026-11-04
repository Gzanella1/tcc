#!/usr/bin/env python3
"""
Gerenciador de e-mails do Gmail via IMAP.

- Não usa Gmail API, Google Cloud, credentials.json ou token.json.
- Usa IMAP com SSL (imap.gmail.com:993).
- Usa e-mail, datas e senha de app definidos na seção CONFIGURAÇÃO.
- Pesquisa mensagens recebidas entre duas datas, incluindo a data final.
- Mostra quantidade e uma pequena amostra.
- Pesquisa mensagens que já estão na Lixeira.
- Só exclui permanentemente após digitar exatamente: EXCLUIR.
"""

from __future__ import annotations

import datetime as dt
import imaplib
import re
import sys
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parsedate_to_datetime

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
IMAP_TIMEOUT = 30
SAMPLE_LIMIT = 10
CONFIRMATION = "EXCLUIR"

# ================================================================
# CONFIGURAÇÃO
# ================================================================
# Estas informações já ficam preenchidas no programa.
EMAIL_CONTA = "giovanizanelladamaia123@gmail.com"
DATA_INICIAL = "01/01/2024"
DATA_FINAL = "30/08/2026"

# A imagem enviada mostra que uma senha de app chamada "gmail" foi
# criada, mas NÃO mostra o código de 16 caracteres da senha.
# Cole aqui o código gerado pelo Google.
#
# IMPORTANTE: ao preencher esta linha, a senha ficará gravada em
# texto puro neste arquivo .py. Não compartilhe esse arquivo.
SENHA_APP = "hxmcmtceczecafhy"


def normalizar_email(valor: str) -> str:
    return valor.strip()


def validar_email(email: str) -> bool:
    # Validação simples apenas para evitar entradas claramente inválidas.
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email))


def ler_data(mensagem: str) -> dt.date:
    while True:
        valor = input(mensagem).strip()
        try:
            return dt.datetime.strptime(valor, "%d/%m/%Y").date()
        except ValueError:
            print("Data inválida. Use exatamente o formato DD/MM/AAAA.\n")


def ler_data_valor(valor: str) -> dt.date:
    """Converte uma data fixa da configuração para DD/MM/AAAA."""
    try:
        return dt.datetime.strptime(valor, "%d/%m/%Y").date()
    except ValueError as exc:
        raise RuntimeError(
            f"Data inválida na configuração: {valor!r}. Use DD/MM/AAAA."
        ) from exc


def formatar_data_imap(data: dt.date) -> bytes:
    # IMAP usa DD-Mon-YYYY nas pesquisas SINCE/BEFORE.
    return data.strftime("%d-%b-%Y").encode("ascii")


def localizar_caixa_especial(mail: imaplib.IMAP4_SSL, atributo: bytes) -> str | None:
    """Localiza uma caixa pelo atributo Special-Use do Gmail, como \\All ou \\Trash."""
    status, respostas = mail.list("", "*")
    if status != "OK":
        return None

    for linha in respostas:
        if not linha:
            continue
        linha_bytes = linha if isinstance(linha, bytes) else linha.encode()
        if atributo not in linha_bytes:
            continue

        # Nomes de mailbox do Gmail são retornados entre aspas neste formato.
        encontrado = re.search(rb'\"((?:[^\"\\]|\\.)*)\"\s*$', linha_bytes)
        if encontrado:
            nome = encontrado.group(1).decode("utf-7", errors="replace")
            return nome

        # Fallback para linhas sem aspas no nome.
        partes = linha_bytes.rsplit(b" ", 1)
        if len(partes) == 2:
            return partes[1].strip(b'\"').decode("utf-7", errors="replace")

    return None


def decodificar_texto_cabecalho(valor: str | None) -> str:
    if not valor:
        return "(sem informação)"
    try:
        return str(make_header(decode_header(valor)))
    except Exception:
        return valor


def formatar_data_cabecalho(valor: str | None) -> str:
    if not valor:
        return "(sem data)"
    try:
        data = parsedate_to_datetime(valor)
        if data.tzinfo:
            return data.astimezone().strftime("%d/%m/%Y %H:%M")
        return data.strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError, OverflowError):
        return valor


def buscar_mensagens(mail: imaplib.IMAP4_SSL, inicio: dt.date, fim: dt.date):
    """
    Procura SOMENTE dentro da Lixeira do Gmail.

    O período usa a data interna do IMAP:
    SINCE = inclusivo
    BEFORE = exclusivo, por isso usamos o dia seguinte à data final.
    """
    trash_mail = localizar_caixa_especial(mail, b"\\Trash")
    if not trash_mail:
        raise RuntimeError("Não foi possível localizar a Lixeira do Gmail via IMAP.")

    status, _ = mail.select(trash_mail, readonly=True)
    if status != "OK":
        raise RuntimeError(
            f"Não foi possível abrir a Lixeira: "
            f"{mail.error.decode(errors='replace')}"
        )

    dia_seguinte = fim + dt.timedelta(days=1)

    status, dados = mail.uid(
        "SEARCH",
        None,
        "SINCE",
        formatar_data_imap(inicio),
        "BEFORE",
        formatar_data_imap(dia_seguinte),
    )

    if status != "OK":
        raise RuntimeError(
            f"Falha na pesquisa IMAP: {mail.error.decode(errors='replace')}"
        )

    uids = dados[0].split() if dados and dados[0] else []
    if not uids:
        return trash_mail, []

    mensagens = []
    parser = BytesParser(policy=policy.default)

    for inicio_chunk in range(0, len(uids), 500):
        chunk = uids[inicio_chunk:inicio_chunk + 500]

        status, dados = mail.uid(
            "FETCH",
            b",".join(chunk),
            "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE MESSAGE-ID)])",
        )

        if status != "OK":
            raise RuntimeError(
                f"Falha ao obter cabeçalhos: "
                f"{mail.error.decode(errors='replace')}"
            )

        for item in dados:
            if not isinstance(item, tuple) or len(item) != 2:
                continue

            metadados, cabecalho_bruto = item
            if not isinstance(metadados, bytes):
                continue
            if not isinstance(cabecalho_bruto, bytes):
                continue

            uid_match = re.search(rb"UID\s+(\d+)", metadados)
            if not uid_match:
                continue

            uid = uid_match.group(1)
            msg = parser.parsebytes(cabecalho_bruto)

            mensagens.append({
                "uid": uid,
                "from": decodificar_texto_cabecalho(msg.get("From")),
                "subject": decodificar_texto_cabecalho(msg.get("Subject")),
                "date": formatar_data_cabecalho(msg.get("Date")),
            })

    mensagens.sort(key=lambda x: x["date"], reverse=True)
    return trash_mail, mensagens


def excluir_permanentemente(
    mail: imaplib.IMAP4_SSL,
    trash_mail: str,
    mensagens,
) -> int:
    """
    EXCLUI PERMANENTEMENTE as mensagens da Lixeira.

    Passos:
    1. Abre a Lixeira.
    2. Marca os UIDs do lote com \\Deleted.
    3. Executa UID EXPUNGE no lote.
    4. Se UID EXPUNGE não estiver disponível, usa EXPUNGE.

    Depois do EXPUNGE, o Gmail não mantém essas mensagens na Lixeira.
    """
    if not trash_mail:
        raise RuntimeError("A Lixeira do Gmail não foi encontrada.")

    status, _ = mail.select(trash_mail, readonly=False)
    if status != "OK":
        raise RuntimeError(
            f"Não foi possível abrir a Lixeira: "
            f"{mail.error.decode(errors='replace')}"
        )

    uids = [m["uid"] for m in mensagens]
    total = len(uids)

    if total == 0:
        return 0

    TAMANHO_LOTE = 50
    excluidos = 0
    falhas = []

    for inicio in range(0, total, TAMANHO_LOTE):
        lote = uids[inicio:inicio + TAMANHO_LOTE]
        fim_lote = inicio + len(lote)

        print(
            f"\rExcluindo permanentemente {fim_lote}/{total}...",
            end="",
            flush=True,
        )

        try:
            uid_set = b",".join(lote)

            # Marca as mensagens do lote para exclusão.
            status, resposta = mail.uid(
                "STORE",
                uid_set,
                "+FLAGS.SILENT",
                r"(\Deleted)",
            )

            if status != "OK":
                falhas.extend(lote)
                print(
                    f"\nFalha ao marcar lote {inicio + 1}-{fim_lote}: "
                    f"{resposta}"
                )
                continue

            # Tenta UID EXPUNGE primeiro.
            try:
                status, _ = mail.uid("EXPUNGE", uid_set)
            except imaplib.IMAP4.error:
                # Alguns servidores/versões IMAP não aceitam UID EXPUNGE.
                # Nesse caso, como acabamos de marcar apenas este lote,
                # EXPUNGE normal remove as mensagens marcadas.
                status, _ = mail.expunge()

            if status == "OK":
                excluidos += len(lote)
            else:
                falhas.extend(lote)

        except (imaplib.IMAP4.error, OSError, TimeoutError) as exc:
            falhas.extend(lote)
            print(f"\nErro no lote {inicio + 1}-{fim_lote}: {exc}")

    print()

    if falhas:
        print("\nATENÇÃO: alguns e-mails não foram excluídos.")
        print(f"Falhas: {len(falhas)}")

    return excluidos


def main() -> int:
    print("=" * 70)
    print("GERENCIADOR DE E-MAILS DO GMAIL — IMAP")
    print("=" * 70)
    print("ATENÇÃO: este programa pode EXCLUIR PERMANENTEMENTE e-mails da Lixeira.\n")

    # Dados definidos na seção CONFIGURAÇÃO acima.
    email_conta = normalizar_email(EMAIL_CONTA)
    senha_app = SENHA_APP.replace(" ", "")

    if not validar_email(email_conta):
        print("Erro: o endereço de e-mail definido no código parece inválido.")
        return 1

    if not senha_app or senha_app == "COLE_AQUI_A_SENHA_DE_APP_DE_16_CARACTERES":
        print("Erro: preencha SENHA_APP no início do código com a senha de app gerada pelo Google.")
        return 1

    inicio = ler_data_valor(DATA_INICIAL)
    fim = ler_data_valor(DATA_FINAL)

    if fim < inicio:
        print("Erro: a data final não pode ser anterior à data inicial.")
        return 1

    mail = None
    try:
        print("\nConectando ao Gmail por IMAP com SSL...")
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, timeout=IMAP_TIMEOUT)

        status, _ = mail.login(email_conta, senha_app)
        senha_app = None  # não mantemos a senha de app depois da autenticação
        if status != "OK":
            print("Erro de autenticação. Confira o e-mail e, principalmente, a senha de aplicativo.")
            return 1

        print("Autenticação concluída.")

        status, capacidades = mail.capability()
        if status != "OK" or not any(
            b"X-GM-EXT-1" in capacidade.upper() for capacidade in capacidades
        ):
            raise RuntimeError("O servidor não anunciou as extensões IMAP específicas do Gmail necessárias pelo programa.")

        # Detecta a Lixeira antes de qualquer alteração.
        trash_mail = localizar_caixa_especial(mail, b"\\Trash")
        if not trash_mail:
            raise RuntimeError("Não foi possível localizar a Lixeira do Gmail via IMAP.")

        trash_mail, mensagens = buscar_mensagens(mail, inicio, fim)

        quantidade = len(mensagens)
        print("\n" + "=" * 70)
        print(f"Período pesquisado: {inicio.strftime('%d/%m/%Y')} até {fim.strftime('%d/%m/%Y')} (inclusive)")
        print(f"E-mails recebidos encontrados: {quantidade}")
        print("=" * 70)

        if quantidade == 0:
            print("Nenhum e-mail correspondente foi encontrado. Nada foi alterado.")
            return 0

        print(f"\nAmostra (até {SAMPLE_LIMIT} e-mails):")
        for numero, mensagem in enumerate(mensagens[:SAMPLE_LIMIT], start=1):
            print(f"\n[{numero}] Data: {mensagem['date']}")
            print(f"    Remetente: {mensagem['from']}")
            print(f"    Assunto:   {mensagem['subject']}")

        print("\n" + "=" * 70)
        print("ATENÇÃO: os e-mails acima ESTÃO NA LIXEIRA.")
        print("A próxima operação apagará esses e-mails PERMANENTEMENTE.")
        confirmacao = input(
            "Digite EXCLUIR para confirmar a exclusão definitiva.\n> "
        )

        if confirmacao != CONFIRMATION:
            print("\nOperação cancelada. Nenhum e-mail foi alterado.")
            return 0

        print("\nConfirmação recebida. EXCLUINDO PERMANENTEMENTE os e-mails da Lixeira...")
        enviados = excluir_permanentemente(mail, trash_mail, mensagens)

        print("\n" + "=" * 70)
        print(f"Operação concluída: {enviados} de {quantidade} e-mails foram excluídos permanentemente.")
        if enviados != quantidade:
            print("Alguns e-mails não foram processados. Eles não foram considerados como enviados para a Lixeira.")
        print("ATENÇÃO: a exclusão permanente foi realizada via EXPUNGE.")
        print("=" * 70)
        return 0

    except imaplib.IMAP4.error as exc:
        print(f"\nErro IMAP/Gmail: {exc}")
        return 1
    except (OSError, TimeoutError) as exc:
        print(f"\nErro de conexão: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n\nOperação interrompida pelo usuário. Nenhuma alteração adicional foi solicitada.")
        return 130
    except Exception as exc:
        print(f"\nErro inesperado: {exc}")
        return 1
    finally:
        if mail is not None:
            try:
                mail.close()
            except Exception:
                pass
            try:
                mail.logout()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
