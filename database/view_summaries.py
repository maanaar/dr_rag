# database/view_summaries.py
# Utility script to view conversation summaries from the database

from conversation_db import get_conversation_summaries, get_conversation_by_id
from typing import Optional
import sys


def view_summaries(limit: int = 10, user_id: Optional[str] = None):
    """View recent conversation summaries"""
    summaries = get_conversation_summaries(limit=limit, user_id=user_id)
    
    if not summaries:
        print("لا توجد ملخصات محادثات في قاعدة البيانات.")
        return
    
    print(f"\n{'='*80}")
    print(f"📋 ملخصات المحادثات ({len(summaries)} ملخص)")
    print(f"{'='*80}\n")
    
    for i, summary in enumerate(summaries, 1):
        print(f"{'─'*80}")
        print(f"📌 الملخص #{i} (ID: {summary['id']})")
        if summary.get('user_id'):
            print(f"   User ID (Meta): {summary['user_id']}")
        print(f"   Session ID: {summary['session_id']}")
        print(f"   التاريخ: {summary['created_at']}")
        print(f"   عدد الرسائل: {summary['message_count']}")
        if summary.get('speciality'):
            print(f"   التخصص المختار: {summary['speciality']}")
        if summary.get('last_bot_response'):
            print(f"\n   آخر رد من البوت:")
            # Truncate if too long
            last_response = summary['last_bot_response']
            if len(last_response) > 200:
                last_response = last_response[:200] + "..."
            print(f"   {last_response}")
        print(f"\n   الملخص:")
        print(f"   {summary['summary']}")
        
        if summary['metadata']:
            metadata = summary['metadata']
            print(f"\n   معلومات إضافية:")
            if metadata.get('has_search'):
                print(f"   - تم البحث عن أطباء")
            if metadata.get('has_booking'):
                print(f"   - تم حجز موعد")
        print()


def view_summary_by_id(record_id: int):
    """View a specific conversation summary by ID"""
    summary = get_conversation_by_id(record_id)
    
    if not summary:
        print(f"❌ لم يتم العثور على ملخص برقم {record_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"📋 ملخص المحادثة (ID: {summary['id']})")
    print(f"{'='*80}\n")
    if summary.get('user_id'):
        print(f"User ID (Meta): {summary['user_id']}")
    print(f"Session ID: {summary['session_id']}")
    print(f"التاريخ: {summary['created_at']}")
    print(f"عدد الرسائل: {summary['message_count']}")
    if summary.get('speciality'):
        print(f"التخصص المختار: {summary['speciality']}")
    if summary.get('last_bot_response'):
        print(f"\n{'─'*80}")
        print("آخر رد من البوت:")
        print(f"{'─'*80}")
        print(summary['last_bot_response'])
    print(f"\n{'─'*80}")
    print("الملخص:")
    print(f"{'─'*80}")
    print(summary['summary'])
    print(f"\n{'─'*80}")
    print("المحادثة الكاملة:")
    print(f"{'─'*80}")
    
    for msg in summary['conversation_data']:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if role == 'user':
            print(f"\n👤 المستخدم: {content}")
        elif role == 'assistant':
            print(f"\n🤖 المساعد: {content}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            record_id = int(sys.argv[1])
            view_summary_by_id(record_id)
        except ValueError:
            print("❌ يرجى إدخال رقم صحيح")
    else:
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        view_summaries(limit=limit)

